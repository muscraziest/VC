#include <cstdlib>
#include <cmath>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <iomanip>
#include <vector>

using namespace cv;
using namespace std;

/****************************************************************************************
Funciones auxiliares
****************************************************************************************/
void mostrarMatriz(Mat m) {

	for (int i=0; i < m.rows; ++i){
		for (int j = 0; j < m.cols; j++)
			cout << setw(15) << setfill(' ') << m.at<double>(i,j) << " ";

		cout << endl;
	}

	cout << endl;
}

/****************************************************************************************
APARTADO 1: ESTIMACIÓN MATRIZ CÁMARA
****************************************************************************************/
//Función para generar una cámara aleatoria
Mat generarCamaraAleatoria(){

	Mat camara = Mat(3,4,CV_32F);
	bool correcta = false;

	while(!correcta){

		//Generamos los valores de la matriz
		for(int i=0; i < 3; ++i)
			for(int j=0; j < 4; ++j)
				camara.at<float>(i,j) = static_cast <float> (rand()+1) / static_cast <float> (RAND_MAX);

		//Tomamos la submatriz para calcular el determinante
		Mat submatriz = Mat(camara, Range::all(), Range(0,3));

		if(determinant(submatriz) != 0)
			correcta = true;
	}

	return camara;
}

//Función para generar puntos del mundo
vector<Mat> generarPuntosMundo(){

	vector<Mat> puntos_mundo;
	Mat m;

	//Creamos los puntos del mundo como matrices columna de 4 componentes (siendo la cuarta siempre 1)
	for(int i=1; i <= 10; ++i){
		for(int j=1; j <= 10; ++j){

			m = Mat(4,1,CV_32F, 1.0);
			m.at<float>(0,0) = 0.0;
			m.at<float>(1,0) = i*0.1;
			m.at<float>(2,0) = j*0.1;
			puntos_mundo.push_back(m);

			m = Mat(4,1,CV_32F, 1.0);
			m.at<float>(0,0) = j*0.1;
			m.at<float>(1,0) = i*0.1;
			m.at<float>(2,0) = 0.0;
			puntos_mundo.push_back(m);

		}
	}
 
	return puntos_mundo;
}

//Función para proyectar puntos 3D por medio de una cámara
vector<Mat> obtenerPuntosProyectados(vector<Mat> puntos, Mat camara){

	vector<Mat> puntos_proyectados;

	//Aplicamos la matriz cámara al punto
	for(int i=0; i < puntos.size(); ++i)
		puntos_proyectados.push_back(camara * puntos.at(i));

	//Homogeneizamos la tercera componente de cada punto
	for(int i=0; i < puntos_proyectados.size(); ++i)
		puntos_proyectados.at(i) = puntos_proyectados.at(i) / puntos_proyectados.at(i).at<float>(2,0);

	return puntos_proyectados;
}

//Función para estimar una cámara a partir de los puntos del mundo y sus proyecciones
Mat estimarCamara(vector<Mat> puntos, vector<Mat> proyecciones){

	Mat A, w, u, vt;
	int f = 2 * puntos.size();
	int c = 12;

	//Estimamos los coeficientes de la matriz
	A = Mat(f,c,CV_32F, 0.0);
	Mat punto_actual, punto_proyectado_actual;

	//Rellenamos la matriz según el esquema que ha de tener al resolver el sistema de ecuaciones
	for(int i=0; i < f; i=i+2){

		punto_actual = puntos.at(i/2);
		punto_proyectado_actual = proyecciones.at(i/2);

		A.at<float>(i,0) = punto_actual.at<float>(0,0);
		A.at<float>(i,1) = punto_actual.at<float>(1,0);
		A.at<float>(i,2) = punto_actual.at<float>(2,0);
		A.at<float>(i,3) = 1.0;

		A.at<float>(i,8) = -punto_proyectado_actual.at<float>(0,0) * punto_actual.at<float>(0,0);
		A.at<float>(i,9) = -punto_proyectado_actual.at<float>(0,0) * punto_actual.at<float>(1,0);
		A.at<float>(i,10) = -punto_proyectado_actual.at<float>(0,0) * punto_actual.at<float>(2,0);
		A.at<float>(i,11) = -punto_proyectado_actual.at<float>(0,0);
  
		A.at<float>(i+1,4) = punto_actual.at<float>(0,0);
		A.at<float>(i+1,5) = punto_actual.at<float>(1,0);
		A.at<float>(i+1,6) = punto_actual.at<float>(2,0);
		A.at<float>(i+1,7) = 1.0;

		A.at<float>(i+1,8) = -punto_proyectado_actual.at<float>(1,0) * punto_actual.at<float>(0,0);
		A.at<float>(i+1,9) = -punto_proyectado_actual.at<float>(1,0) * punto_actual.at<float>(1,0);
		A.at<float>(i+1,10) = -punto_proyectado_actual.at<float>(1,0) * punto_actual.at<float>(2,0);
		A.at<float>(i+1,11) = -punto_proyectado_actual.at<float>(1,0);
	}

	//Obtenemos la descomposición SVD de la matriz de coeficientes
	SVD::compute(A,w,u,vt);

	Mat camara_estimada = Mat(3,4,CV_32F);

	//Construimos la matriz de transformación con la última columna de v (la última fila de vt)
	for(int i=0; i < 3; ++i)
		for(int j=0; j < 4; ++j)
			camara_estimada.at<float>(i,j) = vt.at<float>(11,i*4+j);

	return camara_estimada;
}

//Función para calcular el error de la estimación usando la norma de Frobenius
double calcularErrorEstimacion(Mat A, Mat B){

	Mat diferencia = (A/A.at<float>(0,0)) - (B/B.at<float>(0,0));

	double sum = 0.0;

	for(int i=0; i < diferencia.rows; ++i)
		for(int j=0; j < diferencia.cols; ++j)
			sum += diferencia.at<float>(i,j) * diferencia.at<float>(i,j);

	return sqrt(sum);
}

//Función para obtener las coordenadas píxel de un conjunto de puntos 2D y dibujarlos
void dibujarPuntos(vector<Mat> puntos, Mat imagen, Scalar color,int r){

	float x_max, x_min, y_max, y_min;

	//Calculamos los rangos máximos de valores donde se mueven las coordenadas de los puntos
	x_max = puntos.at(0).at<float>(0,0);
	x_min = puntos.at(0).at<float>(0,0);
	y_max = puntos.at(0).at<float>(1,0);
	y_min = puntos.at(0).at<float>(1,0);

	for(int i=0; i < puntos.size(); ++i){

		if(x_max < puntos.at(i).at<float>(0,0))
			x_max = puntos.at(i).at<float>(0,0);
		else if (x_min > puntos.at(i).at<float>(0,0))
			x_min = puntos.at(i).at<float>(0,0);

		if(y_max < puntos.at(i).at<float>(1,0))
			y_max = puntos.at(i).at<float>(1,0);
		else if (y_min > puntos.at(i).at<float>(1,0))
			y_min = puntos.at(i).at<float>(1,0);
	}

	float longitud_x = x_max - x_min;
	float longitud_y = y_max - y_min;

	//Escalamos para que los puntos puedan verse bien
	float x,y;

	for(int i=0; i < puntos.size(); ++i){
		x = puntos.at(i).at<float>(0,0);
		y = puntos.at(i).at<float>(1,0);

		circle(imagen, Point(ceil((x-x_min)*imagen.cols*longitud_x), ceil((y-y_min)*imagen.rows*longitud_y)), r, color);
	}

}

/****************************************************************************************
APARTADO 2: CALIBRACIÓN DE LA CÁMARA
****************************************************************************************/
//Función para leer las imágenes del tablero
void leerTablero(vector<Mat> &imagenes){

	for(int i=1; i <= 25; ++i){

		char buffer[50];
		sprintf(buffer,"imagenes/Image%d.tif",i);
		imagenes.push_back(imread(buffer,CV_8U));
	}
}

//Función para obtener las esquinas del tablero en las imágenes
void obtenerEsquinas(vector<Mat> imagenes, vector<vector<Point2f> > &esquinas, vector<Mat> &imagenes_calibracion){

	vector<Point2f> esquinas_img_actual;

	for(int i=0; i < 25; ++i){
		if(findChessboardCorners(imagenes.at(i), Size(13,12), esquinas_img_actual)){
			imagenes_calibracion.push_back(imagenes.at(i));
			esquinas.push_back(esquinas_img_actual);
		}

		esquinas_img_actual.clear();
	}

	cout << "Se han localizado todas las esquinas en " << imagenes_calibracion.size() << " imagenes." << endl << endl;

	//Refinamos las coordenadas
	for(int i=0; i < imagenes_calibracion.size(); ++i)
		cornerSubPix(imagenes_calibracion.at(i), esquinas.at(i), Size(5,5), Size(-1,-1), TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,0.1));

	//Pintamos las esquinas encontradas
	for(int i=0; i < imagenes_calibracion.size(); ++i){
		cvtColor(imagenes_calibracion.at(i), imagenes_calibracion.at(i), CV_GRAY2BGR);
		drawChessboardCorners(imagenes_calibracion.at(i), Size(13,12), Mat(esquinas.at(i)), true);
	}
	imshow("Tablero 0", imagenes_calibracion.at(0));
	imwrite("/home/laura/Escritorio/P3/t0.png",imagenes_calibracion.at(0));
	waitKey(0);
  	destroyAllWindows();

	imshow("Tablero 1", imagenes_calibracion.at(1));
	imwrite("/home/laura/Escritorio/P3/t1.png",imagenes_calibracion.at(1));
	waitKey(0);
  	destroyAllWindows();

	imshow("Tablero 2", imagenes_calibracion.at(2));
	imwrite("/home/laura/Escritorio/P3/t2.png",imagenes_calibracion.at(2));
	waitKey(0);
  	destroyAllWindows();

	imshow("Tablero 3", imagenes_calibracion.at(3));
	imwrite("/home/laura/Escritorio/P3/t4.png",imagenes_calibracion.at(3));
  	waitKey(0);
  	destroyAllWindows();
}

//Función para calcular el error sin distorsión óptica
void errorsinDistorsion(Size tam_tablero, vector<Mat> imagenes_calibracion, vector<vector<Point2f> > esquinas_calibracion){

	double error;

	vector<Point3f> esquinas_teoricas;
  
  	//Obtenemos los puntos teoricos donde ha de estar el patron que estamos buscando
  	for( int i=0; i < tam_tablero.height; ++i)
    	for( int j = 0; j < tam_tablero.width; j++)
      		esquinas_teoricas.push_back(Point3f(float(j), float(i), 0));
  
  	//Copiamos los puntos teoricos tantas veces como conjuntos de puntos reales tengamos
  	vector<vector<Point3f> > puntos_objeto;
  	puntos_objeto.resize(imagenes_calibracion.size(), esquinas_teoricas);
  
  	Mat K = Mat::eye(3,3,CV_64F);
  	Mat coef_distorsion = Mat::zeros(8, 1, CV_64F);
  	vector<Mat> rvecs, tvecs;
  
  	error = calibrateCamera (puntos_objeto, esquinas_calibracion, imagenes_calibracion.at(0).size(), K, coef_distorsion, rvecs, tvecs, CV_CALIB_ZERO_TANGENT_DIST|CV_CALIB_FIX_K1|CV_CALIB_FIX_K2|CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5|CV_CALIB_FIX_K6);
  
  	cout << "Los parametros de distorsion y la K calculados suponiendo que no hay ninguna distorsion son: " << endl;
  	for (int i=0; i < 8; ++i)
    	cout << coef_distorsion.at<double>(i,0) << " ";
    
  	cout << endl << endl;
  	cout << "r: " << endl;
  	mostrarMatriz(rvecs.at(0));
  	cout << "t: " << endl;
  	mostrarMatriz(tvecs.at(0));
  	cout << "K: " << endl;
  	mostrarMatriz(K/K.at<double>(0,0));
    
  	cout << endl;
  	cout << "El error con el que se ha calibrado la camara al suponer que no hay distorsion es " << error << "." << endl;
  
}

//Función para calcular el error con distorsión radial
void errorDistorsionRadial(Size tam_tablero, vector<Mat> imagenes_calibracion, vector<vector<Point2f> > esquinas_calibracion){

	double error;

	vector<Point3f> esquinas_teoricas;
  
  	//Obtenemos los puntos teoricos donde ha de estar el patron que estamos buscando
  	for( int i=0; i < tam_tablero.height; ++i)
    	for( int j = 0; j < tam_tablero.width; j++)
      		esquinas_teoricas.push_back(Point3f(float(j), float(i), 0));
  
  	//Copiamos los puntos teoricos tantas veces como conjuntos de puntos reales tengamos
  	vector<vector<Point3f> > puntos_objeto;
  	puntos_objeto.resize(imagenes_calibracion.size(), esquinas_teoricas);
  
  	Mat K = Mat::eye(3,3,CV_64F);
  	Mat coef_distorsion = Mat::zeros(8, 1, CV_64F);
  	vector<Mat> rvecs, tvecs;
  
  	error = calibrateCamera (puntos_objeto, esquinas_calibracion, imagenes_calibracion.at(0).size(), K, coef_distorsion, rvecs, tvecs, CV_CALIB_ZERO_TANGENT_DIST|CV_CALIB_RATIONAL_MODEL);
  
  	cout << "Los parametros de distorsion y la K calculados suponiendo que solo hay distorsion radial son: " << endl;
  	for (int i=0; i < 8; ++i)
    	cout << coef_distorsion.at<double>(i,0) << " ";
  
  	cout << endl << endl;
  	cout << "r: " << endl;
  	mostrarMatriz(rvecs.at(0));
  	cout << "t: " << endl;
  	mostrarMatriz(tvecs.at(0));
  	cout << "K: " << endl;  	
  	mostrarMatriz(K/K.at<double>(0,0));
  
  	cout << "El error al introducir solo distorsion radial es " << error << "." << endl;
}

//Función para calcular el error con distorsión tangencial
void errorDistorsionTangencial(Size tam_tablero, vector<Mat> imagenes_calibracion, vector<vector<Point2f> > esquinas_calibracion){

	double error;

	vector<Point3f> esquinas_teoricas;
  
  	//Obtenemos los puntos teoricos donde ha de estar el patron que estamos buscando
  	for( int i=0; i < tam_tablero.height; ++i)
    	for( int j = 0; j < tam_tablero.width; j++)
      		esquinas_teoricas.push_back(Point3f(float(j), float(i), 0));
  
  	//Copiamos los puntos teoricos tantas veces como conjuntos de puntos reales tengamos
  	vector<vector<Point3f> > puntos_objeto;
  	puntos_objeto.resize(imagenes_calibracion.size(), esquinas_teoricas);
  
  	Mat K = Mat::eye(3,3,CV_64F);
  	Mat coef_distorsion = Mat::zeros(8, 1, CV_64F);
  	vector<Mat> rvecs, tvecs;

  	error = calibrateCamera (puntos_objeto, esquinas_calibracion, imagenes_calibracion.at(0).size(), K, coef_distorsion, rvecs, tvecs, CV_CALIB_FIX_K1|CV_CALIB_FIX_K2|CV_CALIB_FIX_K3|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5|CV_CALIB_FIX_K6);
  
  	cout << "Los parametros de distorsion y la K calculados al suponer que solo hay distorsion tangencial son: " << endl;
  	for (int i=0; i < 8; ++i)
    	cout << coef_distorsion.at<double>(i,0) << " ";
  
  	cout << endl << endl;
  	cout << "r: " << endl;
  	mostrarMatriz(rvecs.at(0));
  	cout << "t: " << endl;
  	mostrarMatriz(tvecs.at(0));
  	cout << "K: " << endl;
  	mostrarMatriz(K/K.at<double>(0,0));
  
  	cout << "El error al introducir solo distorsion tangencial es " << error << "." << endl;
}

//Función para calcular el error con distorsión tangencial
void errorDistorsionRadialTangencial(Size tam_tablero, vector<Mat> imagenes_calibracion, vector<vector<Point2f> > esquinas_calibracion){

	double error;

	vector<Point3f> esquinas_teoricas;
  
  	//Obtenemos los puntos teoricos donde ha de estar el patron que estamos buscando
  	for( int i=0; i < tam_tablero.height; ++i)
    	for( int j = 0; j < tam_tablero.width; j++)
      		esquinas_teoricas.push_back(Point3f(float(j), float(i), 0));
  
  	//Copiamos los puntos teoricos tantas veces como conjuntos de puntos reales tengamos
  	vector<vector<Point3f> > puntos_objeto;
  	puntos_objeto.resize(imagenes_calibracion.size(), esquinas_teoricas);
  
  	Mat K = Mat::eye(3,3,CV_64F);
  	Mat coef_distorsion = Mat::zeros(8, 1, CV_64F);
  	vector<Mat> rvecs, tvecs;

  	error = calibrateCamera (puntos_objeto, esquinas_calibracion, imagenes_calibracion.at(0).size(), K, coef_distorsion, rvecs, tvecs, CV_CALIB_RATIONAL_MODEL);
  
  	cout << "Los parametros de distorsion y la K calculados suponiendo que tenemos ambos tipos de distorsiones son: " << endl;
  	for (int i=0; i < 8; ++i)
    	cout << coef_distorsion.at<double>(i,0) << " ";
  
  	cout << endl << endl;
  	cout << "r: " << endl;
  	mostrarMatriz(rvecs.at(0));
  	cout << "t: " << endl;
  	mostrarMatriz(tvecs.at(0));
  	cout << "K: " << endl;
  	mostrarMatriz(K/K.at<double>(0,0));
  
  	cout << "Calculando todos los coeficientes de distorsion el error es " << error << "." << endl;
}


/****************************************************************************************
APARTADO 3: ESTIMACIÓN DE LA MATRIZ FUNDAMENTAL
****************************************************************************************/

//Función que obtiene los KeyPoints de una imagen con el detector BRISK
vector<KeyPoint> obtenerKeyPoints (Mat im) {

	//Creamos el detector
	Ptr<BRISK> ptrDetectorBRISK = BRISK::create();
	vector<KeyPoint> puntosDetectados;
	
	//Obtenemos los KP:
	ptrDetectorBRISK->detect(im, puntosDetectados);

	return puntosDetectados;
}


//Función que obtiene los descriptores de los KeyPoints localizados mediante un detector BRISK
Mat obtenerDescriptoresBRISK (Mat im) {

	//Creamos el detector:
	Ptr<BRISK> ptrDetectorBRISK = BRISK::create();
	vector<KeyPoint> puntosDetectados;
	Mat descriptores;
	
	//Obtenemos los KP:
	ptrDetectorBRISK->detect(im, puntosDetectados);
	
	//Obtenemos los descriptores para estos KP:
	ptrDetectorBRISK->compute(im, puntosDetectados, descriptores);
	
	return descriptores;
}


//Función que calcula los puntos en correspondencias entre dos imágenes por el criterio de Fuerza Bruta + comprobacion cruzada + BRISK
vector<DMatch> obtenerMatches (Mat im1, Mat im2){
	Mat descriptores1, descriptores2;
	vector<DMatch> matches;
	
	//Creamos el matcher con Fuerza Bruta activandole el flag para el cross check.
	BFMatcher matcher = BFMatcher(NORM_L2, true);
	
	//Obtenemos los descriptores de los puntos obtenidos en cada imagen.
	descriptores1 = obtenerDescriptoresBRISK(im1);
	descriptores2 = obtenerDescriptoresBRISK(im2);
	
	//clock_t t_inicio= clock();
	//Calculamos los matches entre ambas imagenes:
	matcher.match(descriptores1, descriptores2, matches);		
	//printf("FB ha tardado: %.2fs\n",(double)(clock() - t_inicio)/CLOCKS_PER_SEC);
	
	return matches;
}

//Función para calcular la matriz F con el algoritmo de los 8 puntos
Mat obtenerMatrizFundamental(vector<KeyPoint> v1, vector<KeyPoint> v2, vector<Point2f> &c1, vector<Point2f> &c2, vector<DMatch> matches, vector<unsigned char> &buenos_malos){
	
	for (int i=0; i < matches.size(); ++i){
		c1.push_back(v1[matches[i].queryIdx].pt);
		c2.push_back(v2[matches[i].trainIdx].pt);	
	}
	
	//Calculamos la matriz fundamental
	Mat F = findFundamentalMat(c1, c2, CV_FM_8POINT+CV_FM_RANSAC,1,0.99, buenos_malos);
	
	int numero_descartes = 0;
	
	//Vemos cuantas parejas de puntos en correspondencias han sido descartadas por RANSAC
	for (int i=0; i < buenos_malos.size(); ++i)
		if (buenos_malos.at(i) == 0)
			numero_descartes++;
		
	cout << "RANSAC ha descartado " << numero_descartes << " parejas en correspondencias." << endl << endl;

	return F;
}

//Función para calcular y dibujar las líneas epipolares
void dibujarLineasEpipolares(Mat &vmort1, Mat &vmort2, vector<Point2f> &v1, vector<Point2f> &v2, Mat &F, vector<Vec3f> &l1, vector<Vec3f> &l2, vector<unsigned char> &buenos_malos){

	//Calculamos las lineas epipolares para los puntos de cada imagen
	computeCorrespondEpilines(v1, 1, F, l1);
	computeCorrespondEpilines(v2, 2, F, l2);
	
	Vec3f l;
	int pintadas = 0;
	double c = vmort2.cols;
	
	//Dibujamos las lineas epipolares evaluandolas en x = 0 y x = num_columnas_imagen
	for (int i=0; i < l1.size() && pintadas <= 200; ++i) {
		if (buenos_malos.at(i) == 1) {
			l = l1.at(i);
			line(vmort2, Point(0, -l[2]/l[1]), Point(c, (-l[2]-l[0]*c)/l[1]), CV_RGB(rand() % 256,rand() % 256 ,rand() % 256));
			pintadas++;
		}	
	}
	
	c = vmort1.cols;
	pintadas = 0;
	
	for (int i=0; i < l2.size() && pintadas <= 200; ++i) {
		if (buenos_malos.at(i) == 1) {
			l = l2.at(i);
			line(vmort1, Point(0, -l[2]/l[1]), Point(c, (-l[2]-l[0]*c)/l[1]), CV_RGB(rand() % 256,rand() % 256 ,rand() % 256));
			pintadas++;
		}	
	}
	
	imshow("Lineas epipolares de los puntos de Vmort2 sobre Vmort1", vmort1);
	imwrite("/home/laura/Escritorio/P3/l1.png",vmort1);
	waitKey(0);
	destroyAllWindows();	

	imshow("Lineas epipolares de los puntos de Vmort1 sobre Vmort2", vmort2);
	imwrite("/home/laura/Escritorio/P3/l2.png",vmort2);
	waitKey(0);
	destroyAllWindows();
}

//Función para calcular el error medio de la F estimada
void calcularBondadF(vector<Vec3f> l1, vector<Vec3f> l2, vector<Point2f> c1, vector<Point2f> c2, vector<unsigned char> buenos_malos){

	Point2f p;
	Vec3f l;

	//Calculamos el error como las distancia promedio de las lineas epipolares a sus puntos de soporte
	double error1 = 0;
	double error2 = 0;
	int dem = 0;
	for (int i=0; i < l1.size(); ++i) {
		if (buenos_malos.at(i) == 1) {
			l = l1.at(i);
			p = c2.at(i);
			error1 += abs(l[0]*p.x + l[1]*p.y + l[2]) / sqrt(l[0]*l[0]+l[1]*l[1]);
		
			l = l2.at(i);
			p = c1.at(i);
			error2 += abs(l[0]*p.x + l[1]*p.y + l[2]) / sqrt(l[0]*l[0]+l[1]*l[1]);
			
			dem++;
		}
	}
	
	error1 = error1 / dem;
	error2 = error2 / dem;
	
	cout << "El error promedio cometido para las lineas de vmort1 es " << error1 << "." << endl;
	cout << "El error promedio cometido para las lineas de vmort2 es " << error2 << "." << endl;
}

/****************************************************************************************
APARTADO 4: CALCULAR EL MOVIMIENTO DE LA CÁMARA (R,t)
****************************************************************************************/

//Función para estimar la matriz esencial a partir de las matrices F y K
Mat estimarMatrizEsencial(Mat F, Mat K){

	Mat E1 = K.t() * F;
	Mat E = E1 * K;

	return E;
}

//Función para estimar el movimiento de la cámara 
void estimarMovimiento(Mat E, Mat K, vector<Point2f> corresp_1, vector<Point2f> corresp_2){

	//Calculamos E*Et y normalizamos con su traza
	Mat EEt = E*E.t();
	
	double traza = 0.0;
	for (int i=0; i < 3; ++i)
		traza += EEt.at<double>(i,i);

	cout << "Traza de EEtrapuesta: " << traza << endl << endl;
	
	Mat E_norm = E / sqrt(traza/2);

	Mat EEt_norm = E_norm * E_norm.t();
	
	cout << "EEtrapuesta normalizada: " << endl;
	mostrarMatriz(EEt_norm);
	
	//Estimamos la dirección de T
	Mat T = Mat(1,3, CV_64F);
	Mat menos_T = Mat(1,3, CV_64F);
	int fila_donde_despejar;
	
	//Despejamos T de la fila de EEt_norm con el elemento de la diagonal más pequeño
	double elem = EEt_norm.at<double>(0,0);
	for (int i=0; i < 3; ++i)
		if (EEt_norm.at<double>(i,i) <= elem) {
			fila_donde_despejar = i;
			elem = EEt_norm.at<double>(i,i);
		}

	T.at<double>(0, fila_donde_despejar) = sqrt(1-elem);
	T.at<double>(0,(fila_donde_despejar+1)%3) = -EEt_norm.at<double>(fila_donde_despejar, (fila_donde_despejar+1)%3) / sqrt(1-elem);
	T.at<double>(0,(fila_donde_despejar+2)%3) = -EEt_norm.at<double>(fila_donde_despejar, (fila_donde_despejar+2)%3) / sqrt(1-elem);
	
	menos_T.at<double>(0,0) = -T.at<double>(0,0);
	menos_T.at<double>(0,1) = -T.at<double>(0,1);
	menos_T.at<double>(0,2) = -T.at<double>(0,2);
	
	
	//Construimos las rotaciones
	Mat menos_E_norm = -E_norm;
	Mat R_E_T = Mat(3,3,CV_64F);
	Mat R_E_menosT = Mat(3,3,CV_64F);
	Mat R_menosE_T = Mat(3,3,CV_64F);
	Mat R_menosE_menosT = Mat(3,3,CV_64F);
	
	Mat w0 = Mat(1,3, CV_64F);
	Mat w1 = Mat(1,3, CV_64F);
	Mat w2 = Mat(1,3, CV_64F);
	
	Mat R0 = Mat(1,3, CV_64F);
	Mat R1 = Mat(1,3, CV_64F);
	Mat R2 = Mat(1,3, CV_64F);
	
	(E_norm.row(0).cross(T)).copyTo(w0);
	(E_norm.row(1).cross(T)).copyTo(w1);
	(E_norm.row(2).cross(T)).copyTo(w2);
	
	R0 = w0+w1.cross(w2);
	R1 = w1+w2.cross(w0);
	R2 = w2+w0.cross(w1);
	
	(R0).copyTo(R_E_T.row(0));
	(R1).copyTo(R_E_T.row(1));
	(R2).copyTo(R_E_T.row(2));
	
	(E_norm.row(0).cross(menos_T)).copyTo(w0);
	(E_norm.row(1).cross(menos_T)).copyTo(w1);
	(E_norm.row(2).cross(menos_T)).copyTo(w2);
	
	R0 = w0+w1.cross(w2);
	R1 = w1+w2.cross(w0);
	R2 = w2+w0.cross(w1);
	
	(R0).copyTo(R_E_menosT.row(0));
	(R1).copyTo(R_E_menosT.row(1));
	(R2).copyTo(R_E_menosT.row(2));
	
	(menos_E_norm.row(0).cross(T)).copyTo(w0);
	(menos_E_norm.row(1).cross(T)).copyTo(w1);
	(menos_E_norm.row(2).cross(T)).copyTo(w2);
	
	R0 = w0+w1.cross(w2);
	R1 = w1+w2.cross(w0);
	R2 = w2+w0.cross(w1);
	
	(R0).copyTo(R_menosE_T.row(0));
	(R1).copyTo(R_menosE_T.row(1));
	(R2).copyTo(R_menosE_T.row(2));
	
	(menos_E_norm.row(0).cross(menos_T)).copyTo(w0);
	(menos_E_norm.row(1).cross(menos_T)).copyTo(w1);
	(menos_E_norm.row(2).cross(menos_T)).copyTo(w2);
	
	R0 = w0+w1.cross(w2);
	R1 = w1+w2.cross(w0);
	R2 = w2+w0.cross(w1);
	
	(R0).copyTo(R_menosE_menosT.row(0));
	(R1).copyTo(R_menosE_menosT.row(1));
	(R2).copyTo(R_menosE_menosT.row(2));
	
	cout << "La rotacion para E y T es:" << endl;
	mostrarMatriz(R_E_T);
	
	cout << "La rotacion para E y -T es:" << endl;
	mostrarMatriz(R_E_menosT);
	
	cout << "La rotacion para -E y T es:" << endl;
	mostrarMatriz(R_menosE_T);
	
	cout << "La rotacion para -E y -T es:" << endl;
	mostrarMatriz(R_menosE_menosT);
	
	vector<Mat> rotaciones;
	rotaciones.push_back(R_E_T);
	rotaciones.push_back(R_E_menosT);
	rotaciones.push_back(R_menosE_T);
	rotaciones.push_back(R_menosE_menosT);
	
	//Obtenemos la distancia focal en pixels de la matriz de calibración K
	double f = K.at<double>(0,0);
	
	int num_corresp = corresp_1.size();
	double dot1, dot2, Zi, Zd;
	Mat pi=Mat(1,3,CV_64F);
	Mat Pi=Mat(1,3,CV_64F);
	pi.at<double>(0,2) = 1.0;
	
	int R_act = 0;
	Mat R = rotaciones.at(R_act);
	Mat T_act = Mat(1,3,CV_64F);
	T.copyTo(T_act);
	
	int contador = 0;
	bool encontrado = false;
	bool cambio;
	
	//Vemos que combinación es la adecuada
	while (!encontrado) {

		cambio = false;
				
		for (int i=0; i < corresp_1.size() && !cambio && !encontrado; ++i) {
			
			//Calculamos Zi y Zd
			pi.at<double>(0,0) = corresp_1.at(i).x;
			pi.at<double>(0,1) = corresp_1.at(i).y;
			
			dot1 = (f*R.row(0) - corresp_2.at(i).x*R.row(2)).dot(T_act);
			dot2 = (f*R.row(0) - corresp_2.at(i).x*R.row(2)).dot(pi);
			
			Zi=f*dot1/dot2;
			 
			Pi=(Zi/f)*pi;
		
			Zd = R.row(2).dot(Pi-T_act);
			
			//Si ambos negativos cambiamos el signo a T
			if (Zi < 0 && Zd < 0) {
				T_act = -T_act;
				
				if (R_act%2 == 0)
					R_act++;
				else
					R_act--;
					
				R = rotaciones.at(R_act);
				cambio = true;
			}
			//Si tienen signos distintos cambiamos de signo a la E
			else if (Zi*Zd < 0){
				R_act = (R_act+2)%4;
				R = rotaciones.at(R_act);
				cambio = true;
			}
			//Si los dos positivos, hemos acabado.
			else
				encontrado = true;					
		}
	}
	
	cout << "La matriz R es:" << endl;
	mostrarMatriz(R);
	cout << "La matriz T es: " << endl;
	mostrarMatriz(T_act);	

}

/****************************************************************************************
PARTE 1
****************************************************************************************/
void Parte1(){

vector<Mat> imagenes_mosaico, yosemite_full;

	cout << "---------------------------------------------------------------------------------------" << endl;
	cout << "APARTADO 1: ESTIMACIÓN DE LA MATRIZ DE UNA CÁMARA A PATIR DE PUNTOS EN CORRESPONDENCIAS" << endl;
	cout << "---------------------------------------------------------------------------------------" << endl;

	Mat camara_generada = generarCamaraAleatoria();

	vector<Mat> puntos = generarPuntosMundo();

	vector<Mat> proyecciones = obtenerPuntosProyectados(puntos, camara_generada);

	Mat camara_estimada = estimarCamara(puntos, proyecciones);

	vector<Mat> proyecciones_estimada = obtenerPuntosProyectados(puntos, camara_estimada);

	Mat imagen_puntos = Mat::zeros(500,500,CV_32FC3);

	//Dibujamos las proyecciones de ambas cámaras sobre la misma imagen
	dibujarPuntos(proyecciones, imagen_puntos, Scalar(255,0,0),2);

	Mat imagen_proyecciones = imagen_puntos;

	dibujarPuntos(proyecciones_estimada, imagen_puntos, Scalar(0,0,255),1);

	cout << "La camara construida es: " << endl;
	mostrarMatriz(camara_generada);

	cout << endl;

	cout << "La camara estimada es: " << endl;
	mostrarMatriz(camara_estimada);

	cout << endl;

	imshow("Puntos proyectados (rojo) y puntos estimados (azul)", imagen_puntos);
	imwrite("/home/laura/Escritorio/P3/p1.png",imagen_puntos);

	cout << "El error cometido en la aproximación es " << calcularErrorEstimacion(camara_generada, camara_estimada) << "." << endl;

	waitKey(0);
	destroyAllWindows();
}

/****************************************************************************************
PARTE 2
****************************************************************************************/
void Parte2(){

	cout << "---------------------------------------------------------------------------------------" << endl;
	cout << "                APARTADO 2: CALIBRACIÓN DE LA CÁMARA USANDO HOMOGRAFÍAS" << endl;
	cout << "---------------------------------------------------------------------------------------" << endl;

	vector<Mat> imagenes_tablero, imagenes_calibracion;
	vector<vector<Point2f> > esquinas_img_calibracion;

	Size tam_tablero = Size(13,12);

	//Cargamos  las imagenes en color
	leerTablero(imagenes_tablero);

	//Obtenemos las esquinas refinadas
	obtenerEsquinas(imagenes_tablero, esquinas_img_calibracion, imagenes_calibracion);

  	//Calculamos los parametros de calibracion y el error en distintas situaciones
	cout << "----------------------------------------------------------" << endl;
	cout << "Error sin distorsion" << endl;
	cout << "----------------------------------------------------------" << endl;
  	errorsinDistorsion(tam_tablero, imagenes_calibracion, esquinas_img_calibracion);
  	cout << endl;

  	cout << "----------------------------------------------------------" << endl;
	cout << "Error con distorsion radial" << endl;
	cout << "----------------------------------------------------------" << endl;
  	errorDistorsionRadial(tam_tablero, imagenes_calibracion, esquinas_img_calibracion);
  	cout << endl;

  	cout << "----------------------------------------------------------" << endl;
	cout << "Error con distorsion tangencial" << endl;
	cout << "----------------------------------------------------------" << endl;
  	errorDistorsionTangencial(tam_tablero, imagenes_calibracion, esquinas_img_calibracion);
  	cout << endl;

  	cout << "----------------------------------------------------------" << endl;
	cout << "Error con distorsion radial y tangencial" << endl;
	cout << "----------------------------------------------------------" << endl;
  	errorDistorsionRadialTangencial(tam_tablero, imagenes_calibracion, esquinas_img_calibracion); 	
  	cout << endl;
}

/****************************************************************************************
PARTE 3
****************************************************************************************/

//Función donde se estructuran los pasos necesarios para el tercer punto de la practica
void Parte3() {

	cout << "---------------------------------------------------------------------------------------" << endl;
	cout << "                    APARTADO 3: ESTIMACIÓN DE LA MATRIZ FUNDAMENTAL" << endl;
	cout << "---------------------------------------------------------------------------------------" << endl;

	//Cargamos las imagenes
	Mat vmort1 = imread("imagenes/Vmort1.pgm");
	Mat vmort2 = imread("imagenes/Vmort2.pgm");
	
	//Obtenemos puntos clave en las imagenes con BRISK y buscamos matches
	vector<KeyPoint> key_points1 = obtenerKeyPoints(vmort1);
	vector<KeyPoint> key_points2 = obtenerKeyPoints(vmort2);
	vector<DMatch> matches = obtenerMatches(vmort1, vmort2);
	
	cout << "Hemos obtenido " << key_points1.size() << " y " << key_points2.size() << " Key Points en las imágenes Vmort1 y Vmort2 respectivamente." << endl;
	cout << "Y hemos encontrado " << matches.size() << " parejas en correspondencia. " << endl;
	
	//Calculamos F
	vector<Point2f> correspondencias1, correspondencias2;
	vector<unsigned char> buenos_malos;
	Mat F = obtenerMatrizFundamental(key_points1, key_points2, correspondencias1, correspondencias2, matches, buenos_malos);	
	
	cout << "\nLa matriz fundamental es: " << endl;
	mostrarMatriz(F);

	//Dibujamos las líneas epipolares
	vector<Vec3f> lineas_vmort1, lineas_vmort2;
	dibujarLineasEpipolares(vmort1, vmort2, correspondencias1, correspondencias2, F, lineas_vmort1, lineas_vmort2, buenos_malos);
	
	//Calculamos la bondad de F
	calcularBondadF(lineas_vmort1, lineas_vmort2, correspondencias1, correspondencias2, buenos_malos);
}

/****************************************************************************************
PARTE 4
****************************************************************************************/
void Parte4() {

	cout << "---------------------------------------------------------------------------------------" << endl;
	cout << "                 APARTADO 4: CALCULAR EL MOVIMIENTO DE LA CAMARA (R,t)" << endl;
	cout << "---------------------------------------------------------------------------------------" << endl;

	//Cargamos las imagenes
	Mat im0 = imread("imagenes/rdimage.000.ppm");
	Mat im1 = imread("imagenes/rdimage.001.ppm");
	Mat im4 = imread("imagenes/rdimage.004.ppm");
	
	//Obtenemos puntos clave en las imagenes con BRISK y buscamos matches
	vector<KeyPoint> key_points1 = obtenerKeyPoints(im4);
	vector<KeyPoint> key_points2 = obtenerKeyPoints(im1);
	vector<DMatch> matches = obtenerMatches(im4, im1);
	
	//Recuperamos los puntos en correspondencias que se han usado en la estimacion de F
	vector<Point2f> correspondencias1, correspondencias2;
	vector<unsigned char> buenos_malos;
	Mat F = obtenerMatrizFundamental(key_points1, key_points2, correspondencias1, correspondencias2, matches, buenos_malos);	

	cout << "La matriz fundamental es: " << endl;
	mostrarMatriz(F);

	Mat K = Mat(3,3,CV_64F);
	K.at<double>(0,0) = 1839.6300000000001091;
	K.at<double>(0,1) = 0.0;
	K.at<double>(0,2) = 1024.2000000000000455;
	K.at<double>(1,0) = 0.0;
	K.at<double>(1,1) = 1848.0699999999999363;
	K.at<double>(1,2) = 686.5180000000000291;
	K.at<double>(2,0) = 0.0;
	K.at<double>(2,1) = 0.0;
	K.at<double>(2,2) = 1.0;

	cout << "La matriz de parametros intrinsecos es: " << endl;
	mostrarMatriz(K);

	//Estimamos la matriz esencial:
	Mat E = estimarMatrizEsencial(F,K);

	cout << "La matriz esencial E es: " << endl;
	mostrarMatriz(E);
	
	//Estimamos el movimiento
	estimarMovimiento(E, K, correspondencias1, correspondencias2);
}

/****************************************************************************************
MAIN
****************************************************************************************/

int main(int argc, char** argv){

	Parte1();
	cout << endl; 

	Parte2();
	cout << endl;

	Parte3();
	cout << endl;

	Parte4();

	return 0;
}
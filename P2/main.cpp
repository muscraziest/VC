#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

/****************************************************************************************
Funciones auxiliares
****************************************************************************************/

#define PI 3.14159265

struct PuntoHarris{

	float valor;
	Point2f p;
	int escala;
	int orientacion;
};

bool operador(PuntoHarris a, PuntoHarris b){

	return (a.valor > b.valor);
}

//Funcion que lee una imagen desde un archivo y devuelve el objeto Mat donde se almacena
Mat leerImagen(string nombreArchivo, int flagColor = 1){

	//Leemos la imagen con la función imread
	Mat imagen = imread(nombreArchivo,flagColor);

	//Comprobamos si se ha leído la imagen correctamente
	if(!imagen.data)
		cout << "Lectura incorrecta. La matriz esta vacía." << endl;

	return imagen;
}

//Función que muestra una imagen por pantalla
void mostrarImagen(string nombreVentana, Mat &imagen, int tipoVentana = 1){

	//Comprobamos que la imagen no esté vacía
	if(imagen.data){
		namedWindow(nombreVentana,tipoVentana);
		//Mostramos la imagen con la función imshow
		imshow(nombreVentana,imagen);
	}

	else
		cout << "La imagen no se cargó correctamente." << endl;
}

//Función que muestra varias imágenes. Combina varias imágenes en una sola
void mostrarImagenes(string nombreVentana, vector<Mat> &imagenes){

	//Primero calculamos el total de filas y columnas para la imagen que será la unión de todas las imágenes que queramos mostrar
	int colCollage = 0;
	int filCollage = 0;
	int numColumnas = 0;

	for(int i=0; i < imagenes.size(); ++i){
		//Cambiamos la codificación del color de algunas imágenes para evitar fallos al hacer el collage
		if (imagenes[i].channels() < 3) cvtColor(imagenes[i], imagenes[i], CV_GRAY2RGB);
		//Sumamos las columnas
		colCollage += imagenes[i].cols;
		//Calculamos el máximo número de filas necesarias
		if (imagenes[i].rows > filCollage) filCollage = imagenes[i].rows;
	}

	//Creamos la imagen con las dimensiones calculadas y todos los píxeles a 0
	Mat collage = Mat::zeros(filCollage, colCollage, CV_8UC3);

	Rect roi;
	Mat imroi;

	//Unimos todas las imágenes
	for(int i=0; i < imagenes.size(); ++i){

		roi = Rect(numColumnas, 0, imagenes[i].cols, imagenes[i].rows);
		numColumnas += imagenes[i].cols;
		imroi = Mat(collage,roi);

		imagenes[i].copyTo(imroi);

	}

	//Mostramos la imagen resultante
	mostrarImagen(nombreVentana,collage);

}

//Función para reajustar el rango de una matriz al rango [0,255] para poder mostrar correctamente las frecuencias altas
Mat reajustarRango(Mat imagen){

	Mat canales_imagen[3];
	Mat imagen_ajustada;
	Mat canales_ajustada[3];

	//Si la imagen es 1C
	if(imagen.channels() == 1){

		float min = 0;
		float max = 255;

		//Calculamos el rango en el que se mueven los valores de la imagen
		for(int i=0; i < imagen.rows; ++i){
			for(int j=0; j < imagen.cols; ++j){
				if(imagen.at<float>(i,j) < min) min = imagen.at<float>(i,j);
				if(imagen.at<float>(i,j) > max) max = imagen.at<float>(i,j);
			}
		}

		imagen.copyTo(imagen_ajustada);

		for(int i=0; i < imagen_ajustada.rows; ++i)
			for(int j=0; j < imagen_ajustada.cols; ++j)
				imagen_ajustada.at<float>(i,j) = 1.0*(imagen_ajustada.at<float>(i,j)-min)/(max-min)*255.0;


	}

	//Si la imagen es 3C
	else if(imagen.channels() == 3){
		split(imagen,canales_imagen);
		for(int i=0; i < 3; ++i)
			canales_ajustada[i] = reajustarRango(canales_imagen[i]);
		merge(canales_ajustada,3,imagen_ajustada);
	}
	else
		cout << "Número de canales no válido." << endl;

	return imagen_ajustada;
}

//Función para calcular el vector máscara
Mat calcularVectorMascara(float sigma){

	/*Primero calculamos el numero de pixeles que tendrá el vector máscara: a*2+1 para obtener una máscara de orden impar
	Utilizamos round para redondear los números que muestreemos del intervalo [-3sigma, 3sigma], el cual utilizamos para 
	quedarnos con la parte significativa de la gaussiana.
	*/
	int longitud = round(3*sigma)*2+1;
	int centro = (longitud-1)/2; //elemento del centro del vector

	//Calculamos el tamaño del paso de muestreo, teniendo en cuenta que el mayor peso lo va a tener el pixel central 
	//con el valor f(0)
	float paso=6*sigma/(longitud-1);

	//Creamos la imagen que contendrá los valores muestreados
	Mat mascara = Mat(1,longitud,CV_32F);

	//Cargamos los valores en la máscara
	for(int i=0; i <=centro; ++i){
		mascara.at<float>(0,i) = exp(-0.5*(-paso*(centro-1))*(-paso*(centro-1))/(sigma*sigma));
		mascara.at<float>(0,longitud-i-1) = exp(-0.5*(paso*(centro-i))*(paso*(centro-i))/(sigma*sigma));
	}

	//Dividimos por la suma de todos para que los elementos sumen 1
	float suma = 0.0;

	for(int i=0; i < mascara.cols; ++i)
		suma += mascara.at<float>(0,i);

	mascara = mascara /suma;

	return mascara;
}

//Función para calcular un vector preparado para hacer la convolución sin problemas en los píxeles cercanos a los bordes
Mat calcularVectorOrlado(const Mat &senal, Mat &mascara, int cond_contorno){

	//Añadimos a cada lado del vector (longitud_senal -1)/2 píxeles, porque es el máximo número de píxeles que sobrarían
	//al situar la máscara en la esquina.
	Mat copia_senal;

	//Trabajamos con vectores fila
	if(senal.rows == 1)
		copia_senal = senal;
	else if(senal.cols == 1)
		copia_senal = senal.t();
	else
		cout << "El vector senal no es vector fila o columna." << endl;

	int pixel_copia = copia_senal.cols;
	int pixel_extra = mascara.cols-1; //número de píxeles necesarios para orlar
	int cols_vector_orlado = pixel_copia + pixel_extra;

	Mat vectorOrlado = Mat(1,cols_vector_orlado, senal.type());

	int ini_copia, fin_copia; //posiciones donde comienza la copia del vector, centrada

	ini_copia = pixel_extra/2;
	fin_copia = pixel_copia+ini_copia;

	//Copiamos señal centrado en vectorAuxiliar
	for(int i=ini_copia; i < fin_copia; ++i)
		vectorOrlado.at<float>(0,i) = copia_senal.at<float>(0,i-ini_copia);

	//Ahora rellenamos los vectores de orlado. Hacemos el modo espejo.
	for(int i=0; i < ini_copia; ++i){

		vectorOrlado.at<float>(0,ini_copia-i-1) = cond_contorno*vectorOrlado.at<float>(0,ini_copia+i);
		vectorOrlado.at<float>(0,fin_copia+i) = cond_contorno * vectorOrlado.at<float>(0,fin_copia-i-1);
	}

	return vectorOrlado;
}

//Función para calcular la convolución de un vector señal 1D con un canal
Mat calcularConvolucion1D1C(const Mat &senal, Mat &mascara, int cond_contorno){

	//Orlamos el vector para prepararlo para la convolución
	Mat copiaOrlada = calcularVectorOrlado(senal, mascara, cond_contorno);
	Mat segmentoCopiaOrlada;
	Mat convolucion = Mat(1,senal.cols, senal.type());

	int ini_copia, fin_copia, long_lado_orla;
	//Calculamos el rango de píxeles a los que tenemos que aplicar la convolución, excluyen los vectores de orla
	ini_copia = (mascara.cols-1)/2;
	fin_copia = ini_copia + senal.cols;
	long_lado_orla = (mascara.cols-1)/2;

	for(int i=ini_copia; i < fin_copia; ++i){
		//Aplicamos la convolución a cada píxel seleccionado el segmento con el que convolucionamos
		segmentoCopiaOrlada = copiaOrlada.colRange(i-long_lado_orla, i+long_lado_orla+1);
		convolucion.at<float>(0,i-ini_copia) = mascara.dot(segmentoCopiaOrlada);
	}

	return convolucion;
}

//Función para calcular la convolución de una imagen 2D con un sólo canal
Mat calcularConvolucion2D1C(Mat &imagen, float sigma, int cond_bordes){

	//Calculamos el vector máscara
	Mat mascara = calcularVectorMascara(sigma);
	Mat convolucion = Mat(imagen.rows, imagen.cols, imagen.type());

	//Convolución por filas
	for(int i=0; i < imagen.rows; ++i)
		calcularConvolucion1D1C(imagen.row(i),mascara,cond_bordes).copyTo(convolucion.row(i));

	//Convolución por columnas
	convolucion = convolucion.t();//Trasponemos para poder operar como si fuesen filas

	for(int i=0; i < convolucion.rows; ++i)
		calcularConvolucion1D1C(convolucion.row(i),mascara,cond_bordes).copyTo(convolucion.row(i));

	convolucion = convolucion.t();//Deshacemos la trasposición

	return convolucion;
}

//Función que submuestrea una imagen tomando solo las columnas y filas impares
Mat submuestrear1C(const Mat &imagen){

	Mat submuestreado = Mat(imagen.rows/2, imagen.cols/2, imagen.type());

	for(int i=0; i < submuestreado.rows; ++i)
		for(int j=0; j < submuestreado.cols; ++j)
			submuestreado.at<float>(i,j) = imagen.at<float>(i*2+1,j*2+1);

	return submuestreado;
}

//Función que calcula una pirámide Gaussiana
void calcularPiramideGauss(Mat &imagen, vector<Mat> &piramide, int niveles){

	Mat canales_imagen[3];
	Mat canales_nivel[3];
	vector<Mat> canales_piramide[3];

	//Reajustamos el rango de la imagen
	imagen = reajustarRango(imagen);

	//Si la imagen es 1C
	if(imagen.channels() == 1){

		piramide.push_back(imagen);

		for(int i=0; i < niveles-1; ++i){
			piramide.push_back(submuestrear1C(calcularConvolucion2D1C(piramide.at(i),1.5,0)));
		}

	}

	//Si la imagen es 3C
	else if(imagen.channels() == 3){

		piramide.resize(niveles);
		split(imagen,canales_imagen);

		for(int i=0; i < 3; ++i)
			calcularPiramideGauss(canales_imagen[i],canales_piramide[i],niveles);

		for(int i=0; i < niveles; ++i){
			for(int j=0; j < 3; ++j)
				canales_nivel[j] = canales_piramide[j].at(i);

			merge(canales_nivel,3,piramide.at(i));
		}
	}

	else
		cout << "Numero de canales no valido." << endl;
}

//Función para mostrar una pirámide gaussiana
void mostrarPiramide(vector<Mat> piramide, string nombreVentana){

	if(piramide.at(0).channels() == 3){
		for(int i=0; i < piramide.size(); ++i)
			piramide.at(i).convertTo(piramide.at(i),CV_8UC3);
	}

	else if(piramide.at(0).channels() == 1){
		for(int i=0; i < piramide.size(); ++i)
			piramide.at(i).convertTo(piramide.at(i),CV_8U);
	}

	else
		cout << "Numero de canales no valido." << endl;

	mostrarImagenes(nombreVentana,piramide);
}

/****************************************************************************************
APARTADO 1: DETECCIÓN DE PUNTOS HARRIS
****************************************************************************************/

//Función para determinar si un valor es un máximo local en su entorno
bool maximoLocal(Mat entorno, Mat entorno_bin){

	int centro_x = (entorno.rows)/2;
	int centro_y = (entorno.cols)/2;

	bool maximo = true;

	for(int i=0; i < entorno.rows && maximo; ++i){
		for(int j=0; j < entorno.cols && maximo; ++j){
			if(!(i == centro_x && j == centro_y) && entorno_bin.at<float>(i,j) != 0)
				if(entorno.at<float>(centro_x,centro_y) <= entorno.at<float>(i,j))// && entorno_bin.at<float>(i,j) != 0)
					maximo = false;
		}
	}

	return maximo;
}

//Función para cambiar a ceros los valores de una imagen
void modificarCeros(Mat & m, int x, int y, int entorno){

	Mat resultado;

	for(int i=x-entorno/2; i < x+entorno/2+1; ++i){
		for(int j=y-entorno/2; j < y+entorno/2+1; ++j){

			if(!(i == x && j == y))
				m.at<float>(i,j) = 0;
		}
	}
}

//Función para la supresión de no máximos
Mat supresionNoMaximos(Mat harris, int entorno){

	Mat seleccionados = Mat::ones(harris.rows, harris.cols, CV_32FC1)*255;
	int longitud = entorno/2;

	for(int i=longitud; i < harris.rows-longitud; ++i){
		for(int j=longitud; j < harris.cols-longitud; ++j){

			Mat roi = harris(Rect(j-longitud, i-longitud, entorno, entorno));
			Mat roi_bin = seleccionados(Rect(j-longitud, i-longitud, entorno, entorno));

			if(maximoLocal(roi, roi_bin)){
				modificarCeros(seleccionados,i,j,entorno);
			}

			else
				seleccionados.at<float>(i,j) = 0;
		}
	}

	return seleccionados;
}

//Función para calcular el valor de los puntos Harris de una imagen dada
void calcularPuntosHarris (vector<Mat> &piramide, vector<Mat> &harris, vector<PuntoHarris> &p_harris, int escala, int npuntos){

	vector<Mat> derivadas_x, derivadas_y, valores_harris,imagenes_gris;

	for(int i=0; i < piramide.size(); ++i)
		piramide.at(i).convertTo(piramide.at(i),CV_8U);

	//Para cada escala
	for(int i=0; i < escala; ++i){

		//Calculamos los mapas de autovalores
		Mat imagen_gris, harris_aux, dx_aux, dy_aux, abs_dx_aux, abs_dy_aux;
		cvtColor(piramide.at(i), imagen_gris, CV_BGR2GRAY);
		harris_aux = Mat::zeros(piramide.at(i).size(), CV_32FC(6));
		cornerEigenValsAndVecs(imagen_gris, harris_aux, 3, 5, BORDER_DEFAULT);
		imagenes_gris.push_back(imagen_gris);

		//Calculamos los valores Harris
		Mat aux = Mat::zeros(piramide.at(i).size(), CV_32FC1);
		for(int j=0; j < piramide.at(i).rows; ++j){
			for(int k=0; k < piramide.at(i).cols; ++k){
				float lambda1 = harris_aux.at<Vec6f>(j,k)[0];
				float lambda2 = harris_aux.at<Vec6f>(j,k)[1];
				float valor_harris = lambda1 * lambda2 - 0.04f*pow(lambda1+lambda2,2);
				if(valor_harris > 0.001) //<- Ponemos un umbral para evitar ruido
					aux.at<float>(j,k) = valor_harris;
			}
		}

		valores_harris.push_back(aux);

		//Supresión de no máximos
		harris.push_back(supresionNoMaximos(aux, 7));

		//Calculamos las derivadas en x e y con el operador Sobel
		Sobel(imagen_gris, dx_aux, CV_16S, 1, 0, 5);
		convertScaleAbs(dx_aux,abs_dx_aux);
		derivadas_x.push_back(abs_dx_aux);
		Sobel(imagen_gris, dy_aux, CV_16S, 0, 1, 5);
		convertScaleAbs(dy_aux, abs_dy_aux);
		derivadas_y.push_back(abs_dy_aux);
	}

	//Seleccionamos los 1500 puntos de mayor de valor de las distintas escalas
	vector<PuntoHarris> puntos;
	PuntoHarris punto;
	vector<Point2f> puntos_refinados;
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
	
	//Guardamos todos los puntos de todas las escalas
	for(int i=0; i < escala; ++i){
		for(int j=0; j < harris.at(i).rows; ++j){
			for(int k=0; k < harris.at(i).cols; ++k){
				if(harris.at(i).at<float>(j,k) == 255){
					punto.valor = valores_harris.at(i).at<float>(j,k);
					punto.p.x = (float)k*pow(2,i);
					punto.p.y = (float)j*pow(2,i);
					punto.escala = i;
					punto.orientacion = atan(derivadas_y.at(i).at<float>(j,k)/derivadas_x.at(i).at<float>(j,k));
					puntos.push_back(punto);
				}
			}
		}
	}	

	//Ordenamos en orden ascendente
	sort(puntos.begin(), puntos.end(), operador);

	//Guardamos los n primeros
	for(int i=0; i < npuntos; ++i){
		p_harris.push_back(puntos.at(i));
		puntos_refinados.push_back(puntos.at(i).p);
	}

	//Refinamos la posición de los puntos
	for(int i=0; i < p_harris.size(); ++i){

		//Guardamos el punto en un vector auxiliar
		vector<Point2f> p_aux;
		p_aux.push_back(puntos_refinados.at(i));

		//Utilizamos la imagen de la pirámide correspondiente con el punto escogido
		if(p_harris.at(i).escala == 0)
			cornerSubPix(imagenes_gris.at(0), p_aux, Size(5,5), Size(-1,-1), criteria);

		else if(p_harris.at(i).escala == 1)
			cornerSubPix(imagenes_gris.at(1), p_aux, Size(5,5), Size(-1,-1), criteria);

		else if(p_harris.at(i).escala == 2)
			cornerSubPix(imagenes_gris.at(2), p_aux, Size(5,5), Size(-1,-1), criteria);

		else
			cornerSubPix(imagenes_gris.at(3), p_aux, Size(5,5), Size(-1,-1), criteria);

		//Actualizamos el valor de las coordenadas de los puntos
		p_harris.at(i).p.x = p_aux.at(0).x;
		p_harris.at(i).p.y = p_aux.at(0).y;

		p_aux.clear();
	}

}

//Función para mostrar los puntos Harris detectados en una imagen con o sin orientación del punto Harris
void mostrarPuntosHarris(Mat imagen, vector<PuntoHarris> &puntos, bool orientacion){

	float x2, y2;

	//Para cada escala mostramos los puntos de un color y un tamaño distintos
	for(int i=0; i < puntos.size(); ++i){

		if (puntos.at(i).escala == 0){

			//Dibujamos el círculo con centro en las coordenadas del punto de Harris
			circle(imagen,puntos.at(i).p,10*(1+puntos.at(i).escala), Scalar(0,0,255));

			if(orientacion){
				//Dibujamos un segmento que nos indica la dirección del gradiente
				x2 = puntos.at(i).p.x + cos(puntos.at(i).orientacion)*10*(1+puntos.at(i).escala);
				y2 = puntos.at(i).p.y + sin(puntos.at(i).orientacion)*10*(1+puntos.at(i).escala);
				line(imagen,puntos.at(i).p, Point(x2,y2), Scalar(0,0,255));
			}
			
		}

		else if (puntos.at(i).escala == 1){

			//Dibujamos el círculo con centro en las coordenadas del punto de Harris
			circle(imagen,puntos.at(i).p,10*(1+puntos.at(i).escala), Scalar(0,255,0));

			if(orientacion){
				//Dibujamos un segmento que nos indica la dirección del gradiente
				x2 = puntos.at(i).p.x + cos(puntos.at(i).orientacion)*10*(1+puntos.at(i).escala);
				y2 = puntos.at(i).p.y + sin(puntos.at(i).orientacion)*10*(1+puntos.at(i).escala);
				line(imagen,puntos.at(i).p, Point(x2,y2), Scalar(0,255,0));
			}
		}

		else if (puntos.at(i).escala == 2){

			//Dibujamos el círculo con centro en las coordenadas del punto de Harris
			circle(imagen,puntos.at(i).p,10*(1+puntos.at(i).escala), Scalar(255,0,0));

			if(orientacion){
				//Dibujamos un segmento que nos indica la dirección del gradiente
				x2 = puntos.at(i).p.x + cos(puntos.at(i).orientacion)*10*(1+puntos.at(i).escala);
				y2 = puntos.at(i).p.y + sin(puntos.at(i).orientacion)*10*(1+puntos.at(i).escala);
				line(imagen,puntos.at(i).p, Point(x2,y2), Scalar(255,0,0));
			}
		}

		else {

			//Dibujamos el círculo con centro en las coordenadas del punto de Harris
			circle(imagen,puntos.at(i).p,10*(1+puntos.at(i).escala), Scalar(0,255,255));

			if(orientacion){
				//Dibujamos un segmento que nos indica la dirección del gradiente
				x2 = puntos.at(i).p.x + cos(puntos.at(i).orientacion)*10*(1+puntos.at(i).escala);
				y2 = puntos.at(i).p.y + sin(puntos.at(i).orientacion)*10*(1+puntos.at(i).escala);
				line(imagen,puntos.at(i).p, Point(x2,y2), Scalar(0,255,255));
			}
		}		
	}
 
 	if(orientacion)
		mostrarImagen("Puntos de Harris con orientación sobre la imagen original",imagen);	
 	
 	
	else
		mostrarImagen("Puntos de Harris sobre la imagen original",imagen);
	
}

/****************************************************************************************
APARTADO 2: DETECTORES KAZE/AKAZE
****************************************************************************************/

//Función para obtener los KeyPoints del detector KAZE
vector<KeyPoint> obtenerKeyPointKaze(Mat imagen){

	//Creamos el detector
	Ptr<KAZE> ptrKaze = KAZE::create();
	vector<KeyPoint> puntosKaze;
	Mat descriptores;

	//Obtenemos los KeyPoints 
	ptrKaze->detect(imagen, puntosKaze);

	return puntosKaze;

}

//Función para obtener los KeyPoints del detector AKAZE
vector<KeyPoint> obtenerKeyPointAkaze(Mat imagen){

	//Creamos el detector
	Ptr<AKAZE> ptrAkaze = AKAZE::create();
	vector<KeyPoint> puntosAkaze;
	Mat descriptores;

	//Obtenemos los KeyPoints 
	ptrAkaze->detect(imagen, puntosAkaze);

	return puntosAkaze;

}

//Función para obtener los descriptores del detector KAZE
Mat descriptorKaze(Mat imagen){

	//Creamos el detector
	Ptr<KAZE> ptrKaze = KAZE::create();
	vector<KeyPoint> puntosKaze;
	Mat descriptores;

	//Obtenemos los KeyPoints
	ptrKaze->detect(imagen, puntosKaze);

	//Obtenemos los descriptores
	ptrKaze->compute(imagen,puntosKaze,descriptores);

	return descriptores;
	
}

//Función para obtener los descriptores del detector AKAZE
Mat descriptorAkaze(Mat imagen){

	//Creamos el detector
	Ptr<AKAZE> ptrAkaze = AKAZE::create();
	vector<KeyPoint> puntosAkaze;
	Mat descriptores;

	//Obtenemos los KeyPoints
	ptrAkaze->detect(imagen, puntosAkaze);

	//Obtenemos los descriptores
	ptrAkaze->compute(imagen,puntosAkaze,descriptores);

	return descriptores;
	
}

//Función para obtener los matches del detector KAZE
vector<DMatch> obtenerMatchesKaze(Mat imagen1, Mat imagen2){

	Mat descriptor1, descriptor2;
	vector<DMatch> matches;

	//Creamos el matcher con Fuerza Bruta activando el flag de Cross Check
	BFMatcher matcher = BFMatcher(NORM_L2,true);

	//Calculamos los descriptores
	descriptor1 = descriptorKaze(imagen1);
	descriptor2 = descriptorKaze(imagen2);
 
	matcher.match(descriptor1,descriptor2,matches);

	return matches;

}

//Función para obtener los matches del detector AKAZE
vector<DMatch> obtenerMatchesAkaze(Mat imagen1, Mat imagen2){

	Mat descriptor1, descriptor2;
	vector<DMatch> matches;

	//Creamos el matcher con Fuerza Bruta activando el flag de Cross Check
	BFMatcher matcher = BFMatcher(NORM_L2,true);

	//Calculamos los descriptores
	descriptor1 = descriptorAkaze(imagen1);
	descriptor2 = descriptorAkaze(imagen2);

	matcher.match(descriptor1,descriptor2,matches);

	return matches;

}

/****************************************************************************************
APATARDO 3: MOSAICO
****************************************************************************************/

Mat obtenerHomografia(Mat origen, Mat destino){

	vector<KeyPoint> v1, v2;
	vector<DMatch> matches;
	vector<Point2f> p_origen, p_destino;
	Mat homografia;

	//Obtenemos los vectores de KeyPoints
	v1 = obtenerKeyPointAkaze(origen);
	v2 = obtenerKeyPointAkaze(destino);

	//Obtenemos los matches
	matches = obtenerMatchesAkaze(origen, destino);

	//Guardamos los keyPoints como puntos Point2f
	for(int i=0; i < matches.size(); ++i){
		p_origen.push_back(v1[matches[i].queryIdx].pt);
		p_destino.push_back(v2[matches[i].trainIdx].pt);
	}

	//Calculamos la homografía
	homografia = findHomography(p_origen,p_destino,CV_RANSAC);

	homografia.convertTo(homografia, CV_32F);

	return homografia;
}


//Función para crear un mosaico a partir de una lista de imágenes
Mat crearMosaicoN(vector<Mat> imagenes){

  //Creamos la imagen donde formaremos el mosaico
  Mat mosaico = Mat(1000, 2000, imagenes.at(0).type());
  
  //Seleccionamos la posición de la imagen central de la secuencia
  int posicion_central = imagenes.size()/2;
  
  //Colocamos la imagen central del vector en el centro del mosaico
  Mat colocacionCentral = Mat(3,3,CV_32F,0.0);
  
  for (int i = 0; i < 3; i++)
    colocacionCentral.at<float>(i,i) = 1.0;
    
  //Realizamos una traslación para colocarla correctamente
  colocacionCentral.at<float>(0,2) = mosaico.cols/2 - imagenes.at(posicion_central).cols/2;
  colocacionCentral.at<float>(1,2) = mosaico.rows/2 - imagenes.at(posicion_central).rows/2;
  
  warpPerspective(imagenes.at(posicion_central), mosaico, colocacionCentral, Size(mosaico.cols, mosaico.rows), INTER_LINEAR, BORDER_CONSTANT);
  
  //Matrices donde se acumularan las homografías a cada uno de los lados de la imagen central
  Mat h_izq, h_der;

  //Inicializamos con la homografía que hemos calculado antes
  colocacionCentral.copyTo(h_izq);
  colocacionCentral.copyTo(h_der);
  
  //Vamos formando el mosaico empezando desde la imagen central y desplazándonos a ambos extremos calculando las homografías correspondientes
  for (int i = 1; i <= posicion_central; i++) {
    if (posicion_central-i >= 0){
      h_izq = h_izq * obtenerHomografia(imagenes.at(posicion_central-i), imagenes.at(posicion_central-i+1));
      warpPerspective(imagenes.at(posicion_central-i), mosaico, h_izq, Size(mosaico.cols, mosaico.rows), INTER_LINEAR, BORDER_TRANSPARENT); 
    }

    if (posicion_central+i < imagenes.size()){
      h_der = h_der * obtenerHomografia(imagenes.at(posicion_central+i), imagenes.at(posicion_central+i-1));
      warpPerspective(imagenes.at(posicion_central+i), mosaico, h_der, Size(mosaico.cols, mosaico.rows), INTER_LINEAR, BORDER_TRANSPARENT);
    }
  }
  
  	//Eliminamos bordes negros que puedan quedar en la imagen
	vector<Point> p_aux;

	for (int i = 0; i < mosaico.rows; ++i)
		for (int j = 0; j < mosaico.cols; ++j)
			if (mosaico.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
				p_aux.push_back(Point(j, i));
	
	Rect rectangulo = boundingRect(p_aux);
	mosaico=mosaico(rectangulo);

  
  return mosaico;	
}


/****************************************************************************************
PARTE 1
****************************************************************************************/
void Parte1(){

vector<Mat> imagenes_mosaico, yosemite_full;

	Mat yose = imread("./imagenes/Yosemite1.jpg");
	vector<Mat> piramide, harris;
	vector<PuntoHarris> puntos_harris;

	yose.convertTo(yose, CV_32FC1);
	
	calcularPiramideGauss(yose, piramide, 4);

	yose.convertTo(yose, CV_8U);

	//APARTADO 1: PUNTOS DE HARRIS
	cout << "-------------------------------------------------------------" << endl;
	cout << "    APARTADO 1: Detección de puntos Harris en una imagen" << endl;
	cout << "-------------------------------------------------------------" << endl;

	calcularPuntosHarris(piramide,harris,puntos_harris,4,1500);
/*
	cout << "Imagen original: escala 1 " << endl;
	mostrarImagen("Imagen original: escala 1", piramide.at(0));
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();

	cout << "Puntos de harris: escala 1 " << endl;
	mostrarImagen("Puntos de Harris: escala 1", harris.at(0));
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();

	cout << "Imagen original: escala 2" << endl;
	mostrarImagen("Imagen original: escala 2", piramide.at(1));
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();

	cout << "Puntos de harris: escala 2" << endl;
	mostrarImagen("Puntos de Harris: escala 2", harris.at(1));
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();

	cout << "Imagen original: escala 3" << endl;
	mostrarImagen("Imagen original: escala 3", piramide.at(2));
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();

	cout << "Puntos de harris: escala 3 " << endl;
	mostrarImagen("Puntos de Harris: escala ", harris.at(2));
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();

	cout << "Imagen original: escala 4" << endl;
	mostrarImagen("Imagen original: escala 4", piramide.at(3));
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();

	cout << "Puntos de harris: escala 4" << endl;
	mostrarImagen("Puntos de Harris: escala 4", harris.at(3));
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();
*/

	cout << "Puntos de Harris sobre la imagen original" << endl;
	mostrarPuntosHarris(piramide.at(0),puntos_harris,false);
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();

	cout << "Puntos de Harris con orientación sobre la imagen original" << endl;
	mostrarPuntosHarris(piramide.at(0),puntos_harris,true);
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();
}

/****************************************************************************************
PARTE 2
****************************************************************************************/
void Parte2(){

	Mat yose1 = imread("./imagenes/Yosemite1.jpg");
	Mat yose2 = imread("./imagenes/Yosemite2.jpg");
	vector<KeyPoint> puntos_kaze1, puntos_kaze2, puntos_akaze1, puntos_akaze2;
	vector<DMatch> matchKaze, matchAkaze, matchKaze_70, matchAkaze_70;
	Mat imagenKaze, imagenAkaze;
	
	//APARTADO 2: KAZE/AKAZE

	cout << "-------------------------------------------------------------" << endl;
	cout << "        APARTADO 2: Detectores KAZE y AKAZE" << endl;
	cout << "-------------------------------------------------------------" << endl;

	puntos_kaze1 = obtenerKeyPointKaze(yose1);
	puntos_kaze2 = obtenerKeyPointKaze(yose2);
	puntos_akaze1 = obtenerKeyPointAkaze(yose1);
	puntos_akaze2 = obtenerKeyPointAkaze(yose2);

	matchKaze = obtenerMatchesKaze(yose1, yose2);

	for(int i=0; i < 70; ++i)
		matchKaze_70.push_back(matchKaze.at(i));

	cout << "Se han obtenido " << matchKaze.size() << " matches con KAZE." << endl; 

	drawMatches(yose1, puntos_kaze1, yose2, puntos_kaze2, matchKaze_70, imagenKaze, Scalar::all(-1), Scalar::all(-1),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imshow("Matches con detector KAZE",imagenKaze);
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();

	matchAkaze = obtenerMatchesAkaze(yose1, yose2);

	for(int i=0; i < 70; ++i)
		matchAkaze_70.push_back(matchAkaze.at(i));

	cout << "Se han obtenido " << matchAkaze.size() << " matches con AKAZE." << endl;

	drawMatches(yose1, puntos_akaze1, yose2, puntos_akaze2, matchAkaze_70, imagenAkaze, Scalar::all(-1), Scalar::all(-1),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imshow("Matches con detector AKAZE",imagenAkaze);
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();
}

/****************************************************************************************
PARTE 3
****************************************************************************************/
void Parte3(){

	Mat m1 = imread("./imagenes/mosaico002.jpg");
	Mat m2 = imread("./imagenes/mosaico003.jpg");
	Mat m3 = imread("./imagenes/mosaico004.jpg");
	Mat m4 = imread("./imagenes/mosaico006.jpg");
	Mat m5 = imread("./imagenes/mosaico005.jpg");
	Mat m6 = imread("./imagenes/mosaico007.jpg");
	Mat m7 = imread("./imagenes/mosaico008.jpg");
	Mat m8 = imread("./imagenes/mosaico009.jpg");
	Mat m9 = imread("./imagenes/mosaico010.jpg");
	Mat m10 = imread("./imagenes/mosaico011.jpg");
	Mat mosaico;
	vector<Mat> imagenes_mosaico, yosemite_full;

	cout << "-------------------------------------------------------------" << endl;
	cout << "                APARTADO 3: MOSAICO" << endl;
	cout << "-------------------------------------------------------------" << endl;

	imagenes_mosaico.push_back(m1);
	imagenes_mosaico.push_back(m2);
	imagenes_mosaico.push_back(m3);
	imagenes_mosaico.push_back(m4);

//	mosaico = crearMosaicoN(imagenes_mosaico);

//	imshow("Mosaico con 4 imagenes",mosaico);
//	cout << "Pulse una tecla para continuar..." << endl;
//	waitKey(0);
//	destroyAllWindows();

	imagenes_mosaico.push_back(m5);
	imagenes_mosaico.push_back(m6);
	imagenes_mosaico.push_back(m7);

//	mosaico = crearMosaicoN(imagenes_mosaico);

//	imshow("Mosaico con 7 imagenes",mosaico);
//	cout << "Pulse una tecla para continuar..." << endl;
//	waitKey(0);
//	destroyAllWindows();

	imagenes_mosaico.push_back(m8);
	imagenes_mosaico.push_back(m9);
	imagenes_mosaico.push_back(m10);

	mosaico = crearMosaicoN(imagenes_mosaico);

	imshow("Mosaico con 10 imagenes",mosaico);
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();


/*	yosemite_full.push_back(imread("./imagenes/yosemite1.jpg"));
	yosemite_full.push_back(imread("./imagenes/yosemite2.jpg"));
	yosemite_full.push_back(imread("./imagenes/yosemite3.jpg"));
	yosemite_full.push_back(imread("./imagenes/yosemite4.jpg"));

	mosaico = crearMosaicoN(yosemite_full);
	imshow("Yosemite", mosaico);
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();
*/
	cout << "Saliendo..." << endl;
	cout << "-------------------------------------------------------------" << endl;
}

/****************************************************************************************
MAIN
****************************************************************************************/

int main(int argc, char** argv){

	Parte1();

	Parte2();

	Parte3();

	return 0;
}
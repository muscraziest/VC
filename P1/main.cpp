#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

/****************************************************************************************
Funciones auxiliares
****************************************************************************************/

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

/****************************************************************************************
APARTADO A: CONVOLUCIÓN
****************************************************************************************/

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

//Función para calcular la convolución de una imagen 1C o 3C
Mat calcularConvolucion2D(Mat &imagen, float sigma, int cond_bordes){

	Mat convolucion;
	Mat canales[3];
	Mat canalesConvolucion[2];

	//Si la imagen es 1C
	if(imagen.channels() == 1)
		return calcularConvolucion2D1C(imagen, sigma, cond_bordes);

	//Si la imagen es 3C
	else if (imagen.channels() == 3){

		split(imagen,canales);

		for(int i=0; i < 3; ++i)
			canalesConvolucion[i] = calcularConvolucion2D1C(canales[i],sigma,cond_bordes);
		
		merge(canalesConvolucion,3,convolucion);
	}

	else
		cout << "Numero de canales no válido. " << endl;

	return convolucion;
}

/****************************************************************************************
APARTADO B: IMÁGENES HÍBRIDAS
****************************************************************************************/

//Función para calcular una imagen híbrida a partir de dos imágenes dadas
Mat calcularHibrida(Mat &imagen1, Mat &imagen2, float sigma1, float sigma2, Mat &bajas_frec, Mat &altas_frec){

	bajas_frec = calcularConvolucion2D(imagen1,sigma1,0);
	altas_frec = imagen2 - calcularConvolucion2D(imagen2,sigma2,0);

	return bajas_frec + altas_frec;
}

//Función para mostrar la imagen híbrida junto con las frecuencias altas y bajas
void mostrarHibrida(Mat &imagen_hibrida, Mat &bajas_frec, Mat &altas_frec,string nombreVentana){

	//Reajustamos el rango de las imágenes
	imagen_hibrida = reajustarRango(imagen_hibrida);
	altas_frec = reajustarRango(altas_frec);

	//Hacemos la conversión para mostrar las imágenes
	if(imagen_hibrida.channels()==3){

		imagen_hibrida.convertTo(imagen_hibrida,CV_8UC3);
		altas_frec.convertTo(altas_frec,CV_8UC3);
		bajas_frec.convertTo(bajas_frec,CV_8UC3);
	}

	else if(imagen_hibrida.channels() == 1){

		imagen_hibrida.convertTo(imagen_hibrida,CV_8U);
		altas_frec.convertTo(altas_frec,CV_8U);
		bajas_frec.convertTo(bajas_frec,CV_8U);
	}

	else
		cout << "Número de canales no válido." << endl;

	vector<Mat> imagenes;

	imagenes.push_back(altas_frec);
	imagenes.push_back(imagen_hibrida);
	imagenes.push_back(bajas_frec);

	mostrarImagenes(nombreVentana, imagenes);
}


/****************************************************************************************
APARTADO C: PIRÁMIDE GAUSSIANA
****************************************************************************************/

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
MAIN
****************************************************************************************/

int main(int argc, char** argv){

	Mat dog = imread("./imagenes/dog.bmp");
	Mat bird = imread("./imagenes/bird.bmp");
	Mat plane = imread("./imagenes/plane.bmp");

	dog.convertTo(dog, CV_32F);
	bird.convertTo(bird, CV_32FC3);
	plane.convertTo(plane, CV_32FC3);
;
	Mat dog2 = calcularConvolucion2D(dog,3,0);
	Mat dog3 = calcularConvolucion2D(dog,10,0);

	dog.convertTo(dog,CV_8U);
	dog2.convertTo(dog2,CV_8U);
	dog3.convertTo(dog3,CV_8U);

	vector<Mat> apartadoA;

	apartadoA.push_back(dog);
	apartadoA.push_back(dog2);
	apartadoA.push_back(dog3);

	Mat altas, bajas;
	Mat hibrida = calcularHibrida(bird,plane,19.0,1.0,bajas,altas);

	int niveles = 6;
	vector<Mat> piramide;

	calcularPiramideGauss(hibrida,piramide,niveles);

	cout << "Apartado A. Convolución de una imagen para distintos valores de sigma (3.0,10.0)." << endl;
	mostrarImagenes("Apartado A: convolucion",apartadoA);

	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();
	
	cout << "Apartado B. Hibridacion entre un pájaro y un avión." << endl;
	mostrarHibrida(hibrida,altas,bajas,"Apartado B: imagenes hibridas");
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();

	cout << "Apartado C. Piramide Gaussiana." << endl;
	mostrarPiramide(piramide,"Apartado C: piramide gaussiana");
	cout << "Pulse una tecla para continuar..." << endl;
	waitKey(0);
	destroyAllWindows();

	return 0;
}
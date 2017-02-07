//============================================================================
// Name         : proyecto.cpp
// Authors      : Laura Tirado López y Ana Puertas Olea
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <algorithm>
#include <sstream>

using namespace std;
using namespace cv;

bool alisado = false;

//Función auxiliar para convertir un entero a cadena
string convertir_numero(int i){

	stringstream s;
	s << i;
	return s.str();
}


/*Función para detectar la piel
//Utilizamos un filtro de color
//La piel detectada se pondrá en color blanco y lo que no se considera piel se deja en negro
*/
Mat deteccionPiel(Mat imag, int indice){

	Mat imag_hsv;
	string ruta = "/home/alumno/Escritorio/piel/";
	string jpg = ".jpg";
	string nombre = ruta+convertir_numero(indice)+jpg;
	int const max_BINARY_value = 255;

	//Aplicamos un filtro Gaussiano para alisar la imagen y eliminar ruido
	GaussianBlur(imag, imag, Size(3,3),70);

	//Convertimos de RGB a HSV
	cvtColor(imag, imag_hsv, CV_BGR2HSV);
	//Le pasamos la imagen convertida en hsv, y la función deja a negro el píxel que no se encuentra en el umbral definido.
	//En caso de que el píxel se encuentre entre los umbrales establecidos, el píxel se pone a blanco.
	Scalar umbral_bajo = Scalar(0,48,80); //H,S,V
	Scalar umbral_alto = Scalar(179,150,255);

	Mat salida;
	inRange(imag_hsv,umbral_bajo,umbral_alto,salida);
	//imwrite(nombre.c_str(), salida);

	return salida;
}

//Función para recortar la cara
//Buscamos más o menos la posición de la cara y dibujamos un marco negro alrededor
Mat recortarCara(Mat im_piel, Mat original, int indice){

	Mat cara_recortada;
	int x1,x2,y1,y2;
	Vec3b color;
	bool encontrado = false;
	string ruta = "/home/alumno/Escritorio/caras_recortadas/";
	string jpg = ".jpg";
	string nombre = ruta+convertir_numero(indice)+jpg;

	//Buscamos el ancho
	for(int i=0; i < im_piel.cols/2 && !encontrado; ++i){
		color = im_piel.at<Vec3b>(Point(i, im_piel.rows/2));
		if(color.val[0] == 255 && color.val[1] == 255 && color.val[2] == 255){
			encontrado = true;
			x1 = i;
		}
	}

	encontrado = false;
	for(int i=im_piel.cols-1; i > im_piel.cols/2 && !encontrado; --i){
		color = im_piel.at<Vec3b>(Point(i, im_piel.rows/2));
		if(color.val[0] == 255 && color.val[1] == 255 && color.val[2] == 255){
			encontrado = true;
			x2 = i;
		}
	}

	encontrado = false;
	//Buscamos la antura
	for(int i=0; i < im_piel.rows/2 && !encontrado; ++i){
		color = im_piel.at<Vec3b>(Point(im_piel.cols/2, i));
		if(color.val[0] == 255 && color.val[1] == 255 && color.val[2] == 255){
			encontrado = true;
			y1 = i;
		}
	}

	encontrado = false;
	for(int i=im_piel.rows-1; i > im_piel.rows/2 && !encontrado; --i){
		color = im_piel.at<Vec3b>(Point(im_piel.cols/2, i));
		if(color.val[0] == 255 && color.val[1] == 255 && color.val[2] == 255){
			encontrado = true;
			y2 = i;
		}
	}

	//Enmarcamos la cara con un marco negro
	cara_recortada = Mat::zeros(original.rows, original.cols, CV_8UC3);

	for(int i=y1; i < y2; ++i)
		for(int j=x1; j < x2; ++j)
			cara_recortada.at<float>(i,j) = original.at<float>(i,j);

	//imwrite(nombre.c_str(), cara_recortada);

	return cara_recortada;

}

/*Función para detectar los ojos
//Buscamos posibles contornos cerrados en la imagen, buscando formas más o menos circulares
//De todos los posibles candidatos, seleccionamos los que tengan más posibilidades de ser ojos
//comprobando sus posiciones relativas en la imagen
*/
Mat deteccionOjos(Mat imag, int indice){

	RNG rng(12345);
	Mat imagen_contornos, copia, copia_color;
	vector<vector<Point> > contornos;
	vector<Vec4i> jerarquia;
	string ruta = "/home/alumno/Escritorio/contornos/";
	string jpg = ".jpg";
	string nombre;


	//Detectamos la piel y recortamos la cara para facilitar la búsqueda
	Mat piel = deteccionPiel(imag, indice);
	copia_color = recortarCara(piel, imag, indice);
	cvtColor(copia_color,copia,CV_BGR2GRAY);

	//Buscamos los contornos en la imagen
	threshold(copia, imagen_contornos, 85, 255, THRESH_BINARY);
	findContours(imagen_contornos, contornos,  jerarquia, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));

	nombre = "/home/alumno/Escritorio/prep_contornos/";

	//imwrite(nombre+convertir_numero(indice)+jpg, imagen_contornos);

	// Buscamos los rectángulos y elipses adecuados a cada contorno
  	vector<RotatedRect> minRect(contornos.size());
  	vector<RotatedRect> minEllipse(contornos.size());

  	for(int i = 0; i < contornos.size(); i++){
  		minRect[i] = minAreaRect(Mat(contornos[i]));

       	if(contornos[i].size() > 5)
       		minEllipse[i] = fitEllipse(Mat(contornos[i]));
     }


  	// Dibujamos los contornos en una imagen junto con las elipses y rectángulos
  	Mat contornos_finales = Mat::zeros(imagen_contornos.size(), CV_8UC3);
  	for(int i = 0; i< contornos.size(); ++i){

       Scalar color = Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
       drawContours(contornos_finales,contornos,i,color,1,8,vector<Vec4i>(),0,Point());
       ellipse(contornos_finales,minEllipse[i],color,2,8);
       Point2f puntos_rec[4];
       minRect[i].points( puntos_rec);
       for(int j = 0; j < 4; j++)
          line(contornos_finales, puntos_rec[j], puntos_rec[(j+1)%4], color,1,8);
     }

  	nombre = ruta+convertir_numero(indice)+jpg;
  	//imwrite(nombre.c_str(),contornos_finales);

  	// Una vez tenemos todos los candidatos posibles, buscamos los mejores
    Mat salida = imag;
    Point2f centro_ojo1, centro_ojo2, centro_mejor1, centro_mejor2;
	Size2f tam_1, tam_2, tam_mejor1, tam_mejor2;
	int ojo_1, ojo_2, mejor_candidato1, mejor_candidato2;

	mejor_candidato1 = 0;
	mejor_candidato2 = 0;

	centro_mejor1 = minEllipse[mejor_candidato1].center;
	centro_mejor2 = minEllipse[mejor_candidato2].center;
	tam_mejor1 = minEllipse[mejor_candidato1].size;
	tam_mejor2 = minEllipse[mejor_candidato2].size;

	//Recorremos la lista de candidatos y los comprobamos entre ellos
	for(int i=0; i < minEllipse.size()-1; ++i){

		centro_ojo1 = minEllipse[i].center;
		tam_1 = minEllipse[i].size;
		for(int j=0; j < minEllipse.size(); ++j){

			centro_ojo2 = minEllipse[j].center;
			tam_2 = minEllipse[i].size;

			int distancia_x_centro_ojo1 = abs(centro_ojo1.x - imag.cols/2);
			int distancia_x_centro_ojo2 = abs(centro_ojo2.x - imag.cols/2);
			int distancia_y_centro_ojo1 = abs(centro_ojo1.y - imag.rows/2);
			int distancia_y_centro_ojo2 = abs(centro_ojo2.y - imag.rows/2);
			int distancia_x_centro_mejor1 = abs(centro_mejor1.x - imag.cols/2);
			int distancia_x_centro_mejor2 = abs(centro_mejor2.x - imag.cols/2);
			int distancia_y_centro_mejor1 = abs(centro_mejor1.y - imag.rows/2);
			int distancia_y_centro_mejor2 = abs(centro_mejor2.y - imag.rows/2);
			int distancia_y_ojos = abs(centro_ojo1.y-centro_ojo2.y);
			bool candidatos_mitad_superior = centro_ojo1.y <= imag.rows/2 && centro_ojo2.y <= imag.rows/2;
			bool candidatos_mitad_derecha_izquierda = (centro_ojo1.x >= imag.cols/2 && centro_ojo2.x <= imag.cols/2) || (centro_ojo1.x <= imag.cols/2 && centro_ojo2.x >= imag.cols/2);
			bool mejor_candidato1_mas_centrado_x = distancia_x_centro_ojo1 < distancia_x_centro_mejor1;
			bool mejor_candidato2_mas_centrado_x = distancia_x_centro_ojo2 < distancia_x_centro_mejor2;
			bool mejor_candidato1_mas_centrado_y = distancia_y_centro_ojo1 < distancia_y_centro_mejor1;
			bool mejor_candidato2_mas_centrado_y = distancia_y_centro_ojo2 < distancia_y_centro_mejor2;
			bool alturas_similares = distancia_y_ojos <= 5;
			bool misma_altura = centro_ojo2.y == centro_ojo1.y;
			bool distancias_centro_similares = abs(distancia_x_centro_ojo1 - distancia_x_centro_ojo2) <= 100;
			bool areas_similares = abs(tam_1.area()-tam_2.area()) < 5;
			bool area1_correcta = tam_1.area() >= 600 && tam_1.area() <= 3500;
			bool area2_correcta = tam_2.area() >= 600 && tam_2.area() <= 3500;

			//Si los candidatos están a la misma altura por encima de la mitad de la imagen
			//Cada uno en una mitad derecha o izquierda
			//Más o menos a la misma distancia del centro de la imagen
			//Y el área de ambos candidatos es similar los seleccionamos
			if(i != j
			&& candidatos_mitad_superior
			&& candidatos_mitad_derecha_izquierda
			&& mejor_candidato1_mas_centrado_x
			&& mejor_candidato2_mas_centrado_x
			&& mejor_candidato1_mas_centrado_y
			&& mejor_candidato2_mas_centrado_y
			&& (alturas_similares || misma_altura)
			&& distancias_centro_similares
			&& area1_correcta
			&& area2_correcta
			&& areas_similares){

				mejor_candidato1 = i;
				mejor_candidato2 = j;

				centro_mejor1 = minEllipse[mejor_candidato1].center;
				centro_mejor2 = minEllipse[mejor_candidato2].center;
				tam_mejor1 = minEllipse[mejor_candidato1].size;
				tam_mejor2 = minEllipse[mejor_candidato2].size;
			}
		}
	}

	//Dibujamos dos círculos en las zonas seleccionadas para ser los ojos
	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	circle(salida, centro_mejor1,40, Scalar(0,255,0),3);
	circle(salida, centro_mejor2,40, Scalar(0,255,0),3);

	return salida;
}


int main() {

	string ruta_entrada = "/home/alumno/Escritorio/fotos_definitivas/";
	string ruta_salida = "/home/alumno/Escritorio/resultados/";
	string jpg = ".jpg";
	string nombre;
	Mat imagen, resultado;

	for(int numero= 1; numero <= 131; ++numero){

		nombre = ruta_entrada+convertir_numero(numero)+jpg;
		imagen = imread(nombre.c_str(), IMREAD_COLOR);
		resultado = deteccionOjos(imagen,numero);
		nombre = ruta_salida+convertir_numero(numero)+jpg;
		//imwrite(nombre.c_str(), resultado);

	}
	//waitKey(0);
	return 1;
}




#include <iostream>
#include <OpenCV/cv.h>
#include <OPenCV/highgui.h>//Read and display an image
#define _USE_MATH_DEFINES
#include <math.h> 
#include <fstream>


using namespace std;
using namespace cv;


IplImage* images[2];
IplImage* map2D;
IplImage *newImage;
CvCapture* captures[2];
CvFont font, distFont;

CvSeq* contours = 0;
CvMemStorage* contoursStorage = NULL;

ofstream writeFile;
ifstream readFile;

char fileCounter = 0;
char* drive = "H:/MSc_Robotics/Programs/Vision/Images/";
char file[10];
char dest[50];
string chkFilePresent;
bool filePresent = false;
bool firstChk = true;

SurfDescriptorExtractor extractor;
SurfFeatureDetector detector;	

vector<KeyPoint> storedKeypoints, newKeypoints;
Mat storedDescriptors, newDescriptors, img_matches;

FlannBasedMatcher matcher;
Mat H, grayImg;

struct Pose {
	float x;
	float y;	
	float t;
	int d;
};

Pose posFace[100]; //all faces detected
Pose posBody[100];

int offsetX,offsetY;
char nFace=0; 
char nBody=0;
int t1=0,t2=0;
int area[9][9];

int ticks=0;
float robotAngle=0; // degree
float robotX=5000;   //mm
float robotY=5000;	//mm
float **costMat;	//cost matrix

int dis1, dis2; 
int rr1, rr2;
float disparity;
float distance1;
float B = 70;
float F = 3.7;
//float pix2mm = 0.014; //mm
//float rad2degree = 57.2958; //degree
//float Q=0, Fi=0, I=0, J=0, Beta=0, Alpha=0;

/******Face variables*******/
CvHaarClassifierCascade* faceCascades[2];
const char* faceCascade_name ="H:/MSc_Robotics/Programs/Vision/Facetracker/haarcascades/haarcascade_frontalface_alt.xml";
CvMemStorage* faceStorages[2];
CvSeq *faces1,*faces2;

/******Body variables*******/
CvHaarClassifierCascade* bodyCascades[2];
const char* bodyCascade_name ="H:/MSc_Robotics/Programs/Vision/Facetracker/haarcascades/haarcascade_fullbody.xml";
CvMemStorage* bodyStorages[2];
CvSeq *body1,*body2;


int initialize();
int detectFaces();
int detectBody();
int captureFrames();
Pose calculateDistances(int, int, bool);
void displayMap2D();
void detectNearestPoint();
void deleteCostMatrix(int);
int costMatrix(CvSeq*, CvSeq*);
void createCostMatrix(int,int);

void writeDesc(char*);
void readDesc(string);
int recognizer();
void addNewImage(int, int, int, int);

int findMotion(int, bool);
void drawRect(int, int, bool);

int main(int argc, char* argv[])
{
	int key=0;
	int i,j;
	if(initialize()!=0) 
	{
		fprintf( stderr, "xxxxxxxxxxxxx Initialization problems!!! xxxxxxxxxxxxx \n\n");
		//return 0;
	}
		
	while( key != 'q') 
	{
		//t1 = GetTickCount();				

		captureFrames();		
		if (images[0]!=NULL && images[1]!=NULL ) 
		{	
			if(detectFaces()==1)
			{
				if(costMatrix(faces1,faces2)>0)	
				{
					for( i = 0 ; i<faces1->total ; i++ )	
					{
						for( j = 0 ; j<faces2->total ; j++ )
						{
							if (costMat[i][j]==1)
							{										
								if (findMotion(i, false) == 1)
								{					
									drawRect( i, j, true);									
									Pose posf = calculateDistances(i,j, true);								
									if((posf.x)>0 && (posf.x)<10000 && (posf.y)>0 && (posf.y)<10000)
									{
										posFace[nFace]=posf;																		
										nFace++;
									}
								}
							}
						}
					}
				}
			}
			if(detectBody()==1)
			{
				if(costMatrix(body1,body2)>0)
				{					
					for( i = 0 ; i<body1->total ; i++ )		
					{
						for( j = 0 ; j<body2->total ; j++ )
						{
							if (costMat[i][j]==1)
							{		
								if (findMotion(i, true) == 1)
								{
									drawRect( i, j, false);									
									Pose posf = calculateDistances(i,j, false);
									if((posf.x)>0 && (posf.x)<10000 && (posf.y)>0 && (posf.y)<10000)
									{
										posBody[nBody]=posf;																		
										nBody++; 
									}
								}
							}
						}
					}
				}
			}
		}
		displayMap2D();
		key = cvWaitKey(1);
	}	

	
	fprintf( stderr, "\n GOODBYE !!!\n\n");
	//Properly close	
	for(int lr=0;lr<2;lr++){
		cvReleaseCapture(&captures[lr]);
	}	
	if (costMat!=NULL) deleteCostMatrix(faces1->total);
    cvDestroyAllWindows();
	return 0;
}




/**
@Name: initialize
Function who initializes all
*/
inline int initialize()
{
	//fprintf( stderr, "Initialization... \n");
	
	cvInitFont( &font, CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, 1.0,0,1,8 );//Building the background to write the text
	cvInitFont( &distFont, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0.5,0,1,8 );

	/*for(int lr=0;lr<2;lr++)
	{*/
		faceCascades[0] = ( CvHaarClassifierCascade* )cvLoad(faceCascade_name,0,0,0);		// load the classifier	
		bodyCascades[0] = ( CvHaarClassifierCascade* )cvLoad(bodyCascade_name,0,0,0);		// load the classifier	
		faceCascades[1] = ( CvHaarClassifierCascade* )cvLoad(faceCascade_name,0,0,0);		// load the classifier	
		bodyCascades[1] = ( CvHaarClassifierCascade* )cvLoad(bodyCascade_name,0,0,0);		// load the classifier	
		//if( !faceCascades[0] ) 
		// {
		//	fprintf( stderr, "ERROR: Could not load face cascade!!!\n" );
		//	return 2;
		//	//exit(0);
		//}
		//if( !bodyCascades[0] ) 
		//{
		//	fprintf( stderr, "ERROR: Could not load body cascade!!!\n" );
		//	return 2;
		//	//exit(0);
		//}		
		faceStorages[0] = cvCreateMemStorage(0);// setup memory buffer; needed by the face detector
		bodyStorages[0] = cvCreateMemStorage(0);// setup memory buffer; needed by the body detector
        captures[0] = 0;
		images[0] = 0;		
		faceStorages[1] = cvCreateMemStorage(0);// setup memory buffer; needed by the face detector
		bodyStorages[1] = cvCreateMemStorage(0);// setup memory buffer; needed by the body detector
        captures[1] = 0;
		images[1] = 0;	
	//}
	
	 //initialize camera /
	captures[0] = cvCaptureFromCAM(1); //captures[0] = cvCreateCameraCapture(0);
	captures[1] = cvCaptureFromCAM(0);  //captures[1] = cvCreateCameraCapture(1);

	//captures[0] = cvCaptureFromAVI("H:/MSc_Robotics/Programs/Vision/Videos/Malik_Left.avi");
	//captures[1] = cvCaptureFromAVI("H:/MSc_Robotics/Programs/Vision/Videos/Malik_Right.avi");

	
	//Capture setting
	//if( captures[0]  && captures[1])
	//{
	//	fprintf(stderr,"Capture properly initialized...\n");
	//	
 //       for(int i=0;i<2;i++){
	//		cvSetCaptureProperty(captures[i] ,CV_CAP_PROP_FRAME_WIDTH,320);
	//		cvSetCaptureProperty(captures[i] ,CV_CAP_PROP_FRAME_HEIGHT,240);
 //       }
 //   }
	//else
	//{
 //       fprintf(stderr,"ERROR: Could not initialize capturing!!!\n");
	//	return 3;
	//	//exit(0);
 //   }
	
	//init MAP 2D
	map2D=cvCreateImage(cvSize(800,800),8, 3);
	cvZero( map2D );
	cvCircle(map2D, cvPoint(map2D->width/2,map2D->height/2), 10, CV_RGB(255,255,255), -1);
	cvCircle(map2D, cvPoint(map2D->width/2,map2D->height/2), map2D->width/2-1, CV_RGB(255,0,0), 0,2);

	//create windows
	cvNamedWindow("Left",CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Right",CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Map 2D",CV_WINDOW_AUTOSIZE);

	cvMoveWindow( "Left", 5, 5);
	cvMoveWindow( "Right", 350, 5);
	cvMoveWindow( "Map 2D", 610, 130);

	cvShowImage( "Map 2D", map2D);

	//int test= OpenCOM(L"COM3:");//	test= OpenCOM(LPCWSTR("COM3:"));
	//if(test == false) {
	//	fprintf( stderr, "ERROR:  Can't open COM3!!! \n\n");
	//	return 1;
	//	//exit(0);
	//
	//else fprintf(stderr,"Connection properly initialized...\n");

//	I2CInit();

	return 0;
}


/**
@Name: detectFaces
return 1 if faces are detected in the left and in the right
and 0 else
*/
inline int detectFaces()
{
	int i;
	//IplImage* img1;
	//IplImage* img2;	
	// detect faces on image 1 (left)
    faces1 = cvHaarDetectObjects(
            images[0],
            faceCascades[0],
            faceStorages[0],
            1.1,
            2,
            0 /*CV_HAAR_DO_CANNY_PRUNING*/,
            cvSize( 20, 20 ) );
	 // detect faces on image 2 (right)
    faces2 = cvHaarDetectObjects(
            images[1],
            faceCascades[1],
            faceStorages[1],
			1.1,
            2,
            0 /*CV_HAAR_DO_CANNY_PRUNING*/,
            cvSize( 20, 20 ) );

		//img1=cvCloneImage(images[0]);
		//img2=cvCloneImage(images[1]);

		// for each face found, draw a red box 
		for( i = 0 ; i < ( faces1 ? faces1->total : 0 ) ; i++ ) 
		{
			CvRect* r1= ( CvRect* )cvGetSeqElem( faces1, i );
			cvRectangle( images[0],
						cvPoint( r1->x, r1->y ),
						cvPoint( r1->x + r1->width, r1->y + r1->height ),
						CV_RGB( 255, 0, 0 ), 1, 8, 0 );		

			//*****Adding faces into memory******//
			if (recognizer() == 0)
				addNewImage(r1->x+6, r1->y+4, r1->width-8, r1->height-4);
		}
		for( i = 0 ; i < ( faces2 ? faces2->total : 0 ) ; i++ ) 
		{
			CvRect* r2 = ( CvRect* )cvGetSeqElem( faces2, i );
			cvRectangle( images[1],
						cvPoint( r2->x, r2->y ),
						cvPoint( r2->x + r2->width, r2->y + r2->height ),
						CV_RGB( 255, 0, 0 ), 1, 8, 0 );
			//CvPoint center2= cvPoint((r2->x)+(r2->width)/2 ,r2->y+(r2->height)/2);	
		}	
		cvShowImage( "Left", images[0]);
		cvShowImage( "Right",images[1]);
	
	if(faces1->total!=0 && faces2->total!=0) 
		return 1;
	else 
		return 0;   		
}


/**
@Name: detectBody
return 1 if faces are detected in the left and in the right
and 0 else
*/
inline int detectBody()
{
	int i;
	/*IplImage* img1;
	IplImage* img2;	*/
	// detect faces on image 1 (left)
    body1 = cvHaarDetectObjects(
            images[0],
            bodyCascades[0],
            bodyStorages[0],
            1.1,
            2,
            0 /*CV_HAAR_DO_CANNY_PRUNING*/,
            cvSize( 20, 20 ) );
	 // detect faces on image 2 (right)
    body2 = cvHaarDetectObjects(
            images[1],
            bodyCascades[1],
            bodyStorages[1],
			1.1,
            2,
            0 /*CV_HAAR_DO_CANNY_PRUNING*/,
            cvSize( 20, 20 ) );

		/*img1=cvCloneImage(images[0]);
		img2=cvCloneImage(images[1]);
*/
		// for each face found, draw a red box 
		for( i = 0 ; i < ( body1 ? body1->total : 0 ) ; i++ ) 
		{
			CvRect* r1= ( CvRect* )cvGetSeqElem( body1, i );
			cvRectangle( images[0],
						cvPoint( r1->x, r1->y ),
						cvPoint( r1->x + r1->width, r1->y + r1->height ),
						CV_RGB( 255, 0, 0 ), 1, 8, 0 );
			//CvPoint center1= cvPoint((r1->x)+(r1->width)/2 ,r1->y+(r1->height)/2);
		}

		for( i = 0 ; i < ( body2 ? body2->total : 0 ) ; i++ ) 
		{
			CvRect* r2 = ( CvRect* )cvGetSeqElem( body2, i );
			cvRectangle( images[1],
						cvPoint( r2->x, r2->y ),
						cvPoint( r2->x + r2->width, r2->y + r2->height ),
						CV_RGB( 255, 0, 0 ), 1, 8, 0 );
			//CvPoint center2= cvPoint((r2->x)+(r2->width)/2 ,r2->y+(r2->height)/2);	
		}	
		
		cvShowImage( "Left", images[0]);
		cvShowImage( "Right",images[1]);
	
	if(body1->total!=0 && body2->total!=0) 
		return 1;
	else 
		return 0;   		
}


/**
@Name: captureFrames
Function who capture frames
*/
inline int captureFrames()
{
	for(int lr=0;lr<2;lr++)
	{
		//images[lr]=cvRetrieveFrame( captures[lr]); 
		images[lr] = cvQueryFrame(captures[lr]);
		if( !images[lr] ) break;
	}	
	//cvShowImage( "Left", images[0]);
	//cvShowImage( "Right",images[1]);
	return 0;
}


inline Pose calculateDistances(int x, int y, bool isFace)
{
	Pose posFace;
	CvRect* r1;
	CvRect* r2;
	float distancePlan,angleR,angleL,angle,angle1;
	if (isFace == true)
	{
		r1=( CvRect* )cvGetSeqElem(faces1,x);
		r2=( CvRect* )cvGetSeqElem(faces2,y);
	}
	else 
	{
		r1=( CvRect* )cvGetSeqElem(body1,x);
		r2=( CvRect* )cvGetSeqElem(body2,y);
	}
	float d= (float)(r2->x+(r2->width/2) - (r1->x+(r1->width/2)));

	rr1 = (int)(r1->x+(r1->width/2));
	rr2 = (int)(r2->x+(r2->width/2));
	dis1 = 160 - rr1;
	dis2 = 160 - rr2;
	disparity = dis1 - dis2;
	distance1 = (B*F)/ disparity;
	distance1 = abs(distance1);
	//distance1 = ceilf(distance1 * 100) / 100;

	int distanceINT = (distance1 + 0.5) * 10;	

	cout <<"\nDistance:\t" <<distanceINT;

	distancePlan=(-70/d)*34;
	//fprintf( stderr,"DistancePlan: %f\n",distancePlan);
	
	angleR=-atan((r2->x+(r2->width/2)-160)/distancePlan);
	float b=distancePlan/cos(angleR);
	float c=3.5;
	float a= (float)sqrt(b*b+c*c-2*b*c*cos(M_PI/2-angleR));
	angle=(float)acos(distancePlan/a);

	angleL=-atan((r1->x+(r1->width/2)-160)/distancePlan);
	float b1=distancePlan/cos(angleL);	
	float c1=3.5;
	float a1= (float)sqrt(b1*b1+c1*c1-2*b1*c1*cos(M_PI/2-angleL));
	angle1=(float)acos(distancePlan/a1);

	float af=(a+a1)/2;
	float anglef=(((angleR/abs(angleR))*angle)+((angleL/abs(angleL))*angle1))/2;
	
	//printf("Angle:\t%f degree\n\n",anglef/100);
	
	//fprintf( stderr,"Distance: %f\n",af/0.6);
	//fprintf( stderr,"Angle: %f\n",anglef*180/M_PI);
	float xF= (float)- sin(robotAngle*M_PI/180+anglef)*af;
	float yF= (float)- cos(robotAngle*M_PI/180+anglef)*af;
	posFace.x=(robotX+xF*10);
	posFace.y=(robotY+yF*10);
	posFace.t = GetTickCount();
	posFace.d = distanceINT;
	
	detectNearestPoint();
	return posFace;
}


inline void displayMap2D()
{
	
	char dest[20];

	
	cvZero( map2D ); //init image
	cvCircle(map2D, cvPoint((int)(map2D->width/2),(int)(map2D->height/2)), map2D->width/2-1, CV_RGB(255,0,0), 0,2); // large outer circle
	cvCircle(map2D, cvPoint((int)(robotX*map2D->width/10000),(int)(robotY*map2D->height/10000)), 10, CV_RGB(255,255,255), -1); //white circle for the robot
	cvLine(map2D,cvPoint((int)(robotX*map2D->width/10000),(int)(robotY*map2D->height/10000)),cvPoint((int)(robotX/10000*map2D->width-sin(robotAngle*M_PI/180)*10),(int)(robotY/10000*map2D->width-cos(robotAngle*M_PI/180)*10) ),CV_RGB(255,0,0),1,8,0);//red line for the angle of the robot
	
	//cout <<"\n nFace :" <<nFace;
	//cout <<"\n nBody :" <<nBody;
	int objects=0;
	ticks = GetTickCount();
	for(int i=0;i<nFace;i++)
	{
		if(ticks-posFace[i].t < 50000)
		{
			cvCircle(map2D, cvPoint((int)(posFace[i].x*map2D->width/10000),(int)(posFace[i].y*map2D->height/10000)  ), 3, CV_RGB(0,0,255),-1);
			cvCircle(map2D, cvPoint((int)(posFace[i].x*map2D->width/10000),(int)(posFace[i].y*map2D->height/10000)  ), 8, CV_RGB(255,255,255),1);			
			sprintf_s(dest,"%d", posFace[i].d);			
			cvPutText(map2D,dest,cvPoint((int)(posFace[i].x*map2D->width/10000)+8,(int)(posFace[i].y*map2D->height/10000)),&distFont,CV_RGB(255,0,0));
			objects++;
		}
		if(ticks-posFace[i].t > 50000 && ticks-posFace[i].t < 100000)
		{
			cvCircle(map2D, cvPoint((int)(posFace[i].x*map2D->width/10000),(int)(posFace[i].y*map2D->height/10000)  ), 3, CV_RGB(255,255,0),-1);	
			cvCircle(map2D, cvPoint((int)(posFace[i].x*map2D->width/10000),(int)(posFace[i].y*map2D->height/10000)  ), 8, CV_RGB(255,255,0),1);
			sprintf_s(dest,"%d", posFace[i].d);
			cvPutText(map2D,dest,cvPoint((int)(posFace[i].x*map2D->width/10000)+8,(int)(posFace[i].y*map2D->height/10000)),&distFont,CV_RGB(250,255,0));			
			objects++;
		}
	}	
	for(int i=0;i<nBody;i++)
	{
		if(ticks-posBody[i].t < 50000)
		{
			cvCircle(map2D, cvPoint((int)(posBody[i].x*map2D->width/10000),(int)(posBody[i].y*map2D->height/10000)  ), 3, CV_RGB(0,255,0),-1);	
			cvCircle(map2D, cvPoint((int)(posBody[i].x*map2D->width/10000),(int)(posBody[i].y*map2D->height/10000)  ), 11, CV_RGB(255,255,255),1);
			sprintf_s(dest,"%d", posBody[i].d);
			cvPutText(map2D,dest,cvPoint((int)(posBody[i].x*map2D->width/10000)+8,(int)(posBody[i].y*map2D->height/10000)),&distFont,CV_RGB(255,0,0));			
			objects++;
		}
		if(ticks-posBody[i].t > 50000 && ticks-posBody[i].t < 100000)
		{
			cvCircle(map2D, cvPoint((int)(posBody[i].x*map2D->width/10000),(int)(posBody[i].y*map2D->height/10000)  ), 3, CV_RGB(255,255,0),-1);	
			cvCircle(map2D, cvPoint((int)(posBody[i].x*map2D->width/10000),(int)(posBody[i].y*map2D->height/10000)  ), 11, CV_RGB(255,255,0),1);
			sprintf_s(dest,"%d", posBody[i].d);
			cvPutText(map2D,dest,cvPoint((int)(posBody[i].x*map2D->width/10000)+8,(int)(posBody[i].y*map2D->height/10000)),&distFont,CV_RGB(255,255,0));			
			objects++;
		}
	}
	sprintf_s(dest,"%d-Persons",objects);
	cvPutText(map2D,dest,cvPoint(0,30),&font,CV_RGB(250,0,0));

	//cvRectangle( map2D, cvPoint(0,0), cvPoint(map2D->width,map2D->height),CV_RGB(50,0,250),1, 8, 0 );	
	
	cvShowImage( "Map 2D", map2D);

}


inline void detectNearestPoint()
{
	int distanceNearest=10000;
	int pointNearest=-1;
	for(int i=0;i<nFace;i++)	
			pointNearest=i;		
}


inline void deleteCostMatrix(int x)
{
	int i;
	for(i=0;i< x ;i++) free(costMat[i]);  
	free(costMat); 
}


inline int costMatrix(CvSeq *faces1,CvSeq *faces2)
{
	int i,j,x=faces1->total ,y=faces2->total,faceFound=0;
	CvRect* r1;
	CvRect* r2;
	float cost=0.00;
	int offsetMoyX=0,offsetMoyY=0,nOff=0;	

	offsetX=0;
	offsetY=0;

	createCostMatrix(x,y);  //creation of a dynamic matrix

	//find cost
	for( i = 0 ; i<x ; i++ ){
		r1= ( CvRect* )cvGetSeqElem( faces1, i );
		for( j = 0 ; j<y ; j++ ){
			costMat[i][j]=0; // initialization of the matrix to zero
			r2= ( CvRect* )cvGetSeqElem(faces2, j );
			
			float costY=((float)(images[0]->height)-(float)abs(r1->y-r2->y))/(float)(images[0]->height); 			
			if (costY<0.95) costY=0;//need to be >0,95
			float costWidth=((float)(images[0]->width)-(float)abs(r1->width-r2->width))/(float)(images[0]->width);
			if (costWidth<0.97) costWidth=0;
			float costHeight=((float)(images[0]->height)-(float)abs(r1->height-r2->height))/(float)(images[0]->height);
			if (costHeight<0.97) costHeight=0;
			
			cost=(float) (costY+costWidth*0.5+costHeight*0.5)/2;
			if (cost<0.95) cost=0;
			else cost=1;
			costMat[i][j]=cost;
		}
    }

	//found offset
	int k=0,line=0;
	float value=0;
	for( i = 0 ; i<x ; i++ ){
		k=0;
		for( j = 0 ; j<y ; j++ ){
			if (costMat[i][j]!=0) k++,line=j;
			if(k>1) break;
		}
		if (k==1){
			//peut-etre inutile!
			value=costMat[i][line];
			for( k = 0 ; k<x ; k++ )costMat[k][line]=0;
			costMat[i][line]=value;
			
			
			r1=( CvRect* )cvGetSeqElem( faces1, i );
			r2=( CvRect* )cvGetSeqElem(faces2, line );
			offsetMoyX+=(r1->x-r2->x);			
			offsetMoyY+=(r1->y-r2->y);
			nOff++;	
			//break;
		}
	}
	if(nOff!=0){
		offsetX=offsetMoyX/nOff;
		offsetY=offsetMoyY/nOff;
	
	
		//Correct the cost matrix with offset
		for( i = 0 ; i<x ; i++ )
			for( j = 0 ; j<y ; j++ )
				if (costMat[i][j]!=0){
					r1=( CvRect* )cvGetSeqElem( faces1,i);
					r2=( CvRect* )cvGetSeqElem(faces2,j);
					if (abs(r1->x-r2->x-offsetX)>10) costMat[i][j]=0;
					else costMat[i][j]=1;
				}
	}
	//check the matrix, to ensure there are no more than one value per line				
	for( i = 0 ; i<x ; i++ ){
		k=0;
		for( j = 0 ; j<y ; j++ )
			if (costMat[i][j]!=0) k++;
		if(k>1){
			for( j = 0 ; j<y ; j++ )costMat[i][j]=0; //whole column to zero
		}
		if(k==1) faceFound++;
	}

	if (faceFound>0){   //Printf costMat
		/*
		
		fprintf( stderr, "\nOffsetX: %d   offsetY: %d \n ",offsetX, offsetY);
		fprintf( stderr, "Cost matrix: \n");//fprintf( stderr, "Mat[%d][%d]=%f\n",i,j,costMat[i][j]);
		for( j = 0 ; j<y ; j++ ){
			for( i = 0 ; i<x ; i++ )
				fprintf( stderr, "%.2f ",costMat[i][j]);
			fprintf( stderr, "\n");
		}
		*/		
		return faceFound;	
	}
	else return 0;
	
}


inline void createCostMatrix(int x,int y)
{
	int i;	
	costMat = (float**)malloc( x * sizeof(float*));
	if( costMat == NULL )
	{
		 fprintf(stderr,"ERROR: Allocation error 1");
		 exit(EXIT_FAILURE);
	}
	for( i = 0 ; i < x ; i++ )
	{
		 costMat[i] = (float*)malloc (y*sizeof(float));	     
		 if( costMat[i] == NULL )
		 {
			 fprintf(stderr,"ERROR: Allocation error 2: %d",i);
			  exit(EXIT_FAILURE);
		 }
	}

}




///********* Face recognition part ***************/////


inline void addNewImage(int x, int y, int w, int h)
{
	//converting Mat obj to IplImage
	newImage = cvCreateImage( cvSize( images[0]->width, images[0]->height ),
		images[0]->depth, images[0]->nChannels );	
	/* copy from source to dest */
	cvCopy( images[0], newImage, NULL );

	//extracting new image from whole window
	cvSetImageROI(newImage, cvRect(x, y, w, h));						
	
	//cvShowImage("Extracted Face", newImage);

	sprintf(file, "face_%d", fileCounter);	
	sprintf(dest,"%s%s.jpg", drive, file);
	cvSaveImage(dest,newImage);	

	writeDesc(file);

	fileCounter++;
	if (fileCounter > 9)
		fileCounter=0;
}


inline int recognizer()
{	
	vector<vector<DMatch > > matches;
	vector<DMatch > good_matches;

	Mat srcImg(images[0]);
	Mat srcCopy = srcImg.clone();
	Mat grayImg;
	Mat img_matches;

	cvtColor(srcImg, grayImg, CV_RGB2GRAY);

	detector.detect( grayImg, newKeypoints );
	extractor.compute( grayImg, newKeypoints, newDescriptors );

	for(int i=0; i<10; i++)
	{		
		sprintf(file, "face_%d", i);	
		sprintf(dest,"%sface_%d.txt", drive, i);
		try
		{	
			readDesc(dest);				
			matcher.knnMatch(newDescriptors, storedDescriptors, matches, 2);		
		}
		catch(Exception e){continue;}

		for(int i = 0; i < min(newDescriptors.rows-1,(int) matches.size()); i++) 
		{
			if((matches[i][0].distance < 0.6*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
			{
				good_matches.push_back(matches[i][0]);
			}
		}

		if (good_matches.size() >= 3)
		{			
			putText( srcImg, file, Point(50,50), CV_FONT_HERSHEY_COMPLEX, 1,
					Scalar(255, 0, 0), 1, 8 );			
			//imshow("Face", srcImg);
			good_matches.clear();
			return 1;
		}	
		//else					
			//imshow("Face", srcCopy);		
	}	
	good_matches.clear();
	return 0;
}
	
inline void writeDesc(char* file)
{	
	sprintf(dest,"%s%s.txt", drive, file);
	string fileLocation(dest);

	Mat srcImg(newImage);	
	storedKeypoints.clear();
	cvtColor(srcImg, grayImg, CV_RGB2GRAY);
			
	detector.detect( grayImg, storedKeypoints );
	extractor.compute( grayImg, storedKeypoints, storedDescriptors );	
	
	FileStorage fs2(fileLocation, FileStorage::WRITE);	
	fs2 << "f" << storedDescriptors;	
}

inline void readDesc(string fileLocation)
{
	FileStorage fs2(fileLocation,FileStorage::READ);
	fs2["f"] >> storedDescriptors;
}



///********* Motion detection part ***************/////


inline int findMotion(int face, bool isBody)
{
	CvRect *r1;
	CvRect rect;
	//create storage for contours
	contoursStorage = cvCreateMemStorage(0);

	IplImage *differenceImg, *oldFrame_grey, *currentFrame_grey;
	
	CvSize size;
	size.width = images[0]->width;
	size.height = images[0]->height;

	IplImage *contourImg = cvCreateImage(size, images[0]->depth, 3);
	currentFrame_grey = cvCreateImage( size, IPL_DEPTH_8U, 1);
	oldFrame_grey = cvCreateImage( size, IPL_DEPTH_8U, 1);
	differenceImg = cvCreateImage( size, IPL_DEPTH_8U, 1);

	contourImg = images[0];
	cvCvtColor(images[0], oldFrame_grey,CV_RGB2GRAY);
	captureFrames();
	cvCvtColor(images[0],currentFrame_grey,CV_RGB2GRAY);

	//Minus the current frame from the moving average.
	cvAbsDiff(oldFrame_grey,currentFrame_grey,differenceImg);

	//bluring the differnece image
	cvSmooth(differenceImg, differenceImg, CV_BLUR);		

	//apply threshold to discard small unwanted movements
	cvThreshold(differenceImg, differenceImg, 50, 255, CV_THRESH_BINARY);

	//find contours
	cvFindContours( differenceImg, contoursStorage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL );		

	if(isBody)
		r1=( CvRect* )cvGetSeqElem(body1,face);
	else if(isBody == false)
		r1=( CvRect* )cvGetSeqElem(faces1,face);	

	for(; contours!=0; contours = contours->h_next)
	{
		rect = cvBoundingRect(contours, 0); //extract bounding box for current contour
		
		//drawing rectangle
		cvRectangle(contourImg,                   
				cvPoint(rect.x, rect.y),     
				cvPoint(rect.x+rect.width, rect.y+rect.height),
				cvScalar(0, 255, 0, 0), 
				2, 8, 0);	
				
		if (rect.x >= r1->x && rect.x <= r1->x+r1->width && rect.y >= r1->y && rect.y <= r1->y+r1->height)					
		{
			cvShowImage("con@", contourImg);
			return 1;		
		}
		else if (rect.x+rect.width >= r1->x && rect.x+rect.width <= r1->x+r1->width && rect.y+rect.height >= r1->y && rect.y+rect.height <= r1->y+r1->height)						
		{
			cvShowImage("con@", contourImg);
			return 1;		
		}
	}		
	return 0;
}

void drawRect(int i, int j, bool isFace)
{
	CvRect* r1, *r2;
	if (isFace == true)
	{
		r1= ( CvRect* )cvGetSeqElem( faces1, i );
		r2 = ( CvRect* )cvGetSeqElem( faces2, j );
	}
	if (isFace == false)
	{
		r1= ( CvRect* )cvGetSeqElem( body1, i );
		r2= ( CvRect* )cvGetSeqElem( body2, j );		
	}			

	cvRectangle( images[0],
					cvPoint( r1->x+4, r1->y+4 ),
					cvPoint( r1->x + r1->width-4, r1->y + r1->height-4 ),
					CV_RGB( 0, 0, 255), 1, 8, 0 );		
	cvRectangle( images[1],
				cvPoint( r2->x+4, r2->y+4 ),
				cvPoint( r2->x + r2->width-4, r2->y + r2->height-4 ),
				CV_RGB( 0, 0, 255), 1, 8, 0 );

	cvShowImage("Left", images[0]);
	cvShowImage("Right", images[1]);
}
/**********************************************************************************/
/* Copyright (c) 2018, Christopher Deotte                                         */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files (the "Software"), to deal  */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/stat.h>
#include<unistd.h>
#include<dirent.h>
#include<math.h>
#include<time.h>
#include<pthread.h>

/* WEBGUI External Routines needed from webgui.c */
/* (these can be placed into a webgui.h) */

int webstart(int x);
void webreadline(char* str);
void webwriteline(char* str);
void webinit(char* str,int x);
void webupdate(int* ip, double* rp, char* sp);
void websettitle(char* str);
void websetmode(int x);
void webstop();
void websetcolors(int nc, double* r,double* g,double* b,int pane);
void webimagedisplay(int nx, int ny, int* image, int ipane);
void webframe(int frame);
void weblineflt(float* x, float* y, float* z, int n, int color);
void webfillflt(float* x, float* y, float* z, int n, int color);
void weblinedbl(double* x, double* y, double* z, int n, int color);
void webfilldbl(double* x, double* y, double* z, int n, int icolor);
void webgldisplay(int pane);
int webquery();
void webbutton(int highlight,char* cmd);
void webpause();
unsigned long fsize(char* file);

/* WEBGUI Internal Routines and Variables */
/* general routines for any program using webgui.c */

void initParameterMap(char* str, int n);
char* extractVal(char* str, char key);
char processCommand(char* str);
void updateParameter(char* str, int index1, int index2);
char arrayGet(char* key);
int ipGet(char* key);
void ipSet(char* key, int value);
double rpGet(char* key);
void rpSet(char* key, double value);
char* spGet(char* key);
void spSet(char* key, char* value);
/* general variables for any program using webgui.c */
int ct=0;
char** map_keys;
int* map_indices;
char* map_array;
double *rp_default, *rp;
int *ip_default, *ip;
char *sp_default, *sp;
char buffer[80];

/* Program Specific Routines */
/* organized by category     */

// LOAD DATA
int loadTrain(int ct, double testProp, int sh, float imgScale, float imgBias);
int loadTest(int ct, int sh, int rc, float imgScale, float imgBias);
// DISPLAY DATA
void displayDigit(int x, int ct, int p, int lay, int chan, int train, int cfy, int big);
void displayDigits(int *dgs, int ct, int pane, int train, int cfy, int wd, int big);
void doAugment(int img, int big, int t);
void viewAugment(int img, int ct, float rat, int r, float sc, int dx, int dy, int p, int big, int t, int cfy);
void printDigit(int a,int b,int *canvas,int m,int n,float *digit,int row,int col,int d3,int d1,int d2);
void printInt(int a,int b,int *canvas,int m,int n,int num,int col,int d1);
void printStr(int a,int b,int *canvas,int m,int n,char *str,int col,int d1);
// DISPLAY PROGRESS
void displayConfusion(int (*confusion)[10]);
void displayCDigits(int x,int y);
void displayEntropy(float *ents, int entSize, float *ents2, int display);
void displayAccuracy(float *accs, int accSize,float *accs2, int display);
void line(int* img, int m, int n,float x1, float y1, float x2, float y2,int d,int c);
// DISPLAY DOTS
void clearImage(int p);
void updateImage();
void displayClassify(int dd);
void placeDots();
void removeDot(float x, float y);
// INIT-NET
void initNet(int t);
void initArch(char *str, int x);
// NEURAL-NET
int isDigits(int init);
void randomizeTrainSet();
void dataAugment(int img, int r, float sc, float dx, float dy, int p, int hiRes, int loRes, int t);
void *runBackProp(void *arg);
int backProp(int x,float *ent, int ep);
int forwardProp(int x, int dp, int train);
float ReLU(float x);
float TanH(float x);
// KNN
void *runKNN(void *arg);
int singleKNN(int x, int k, int d, int p, int train, int big, int disp);
void fakeHeap(float *dist,int *idx,int k);
void sortHeap(float *dist,int *idx,int k);
float distance(float *digitA, float *digitB, int n, int x);
// PREDICT
void *predictKNN(void *arg);
void writePredictFile(int NN, int k, int d, int y, int big);
void writeFile();
// SPECIAL AND/OR DEBUG
void dreamProp(int y, int it, float bs, int ds);
void dream(int x, int y, int it, float bs, int ds);
void displayWeights();
void writeFile2();
void initData(int z);
void boundingBoxes();
float square(float x);

/* Program Specific Variables */

// MENU CREATION ARRAY
// explained in webgui.c user manual
//
int lines = 200;
char init[200][80]={
    "c c=Load, k=l",
    "c c=Display, k=p",
    "c c=Init-Net, k=i",
    "c c=Activation, k=a",
    "c c=DropOut, k=o",
    "c c=DataAugment, k=d",
    "c c=Train-Net, k=b",
    "c c=Find-kNN, k=g",
    "c c=Validate-kNN, k=k",
    "c c=STOP, k=t",
    "c c=Predict, k=w",
    "c c=QUIT, k=q",
    "c c=DotColor, k=s",
    "c c=ClearPane3, k=c",
    "c c=DotRemoveMode, k=r",
    "c c=DotLowResMode, k=u",
// Special features and/or debugging
//    "c c=Dream, k=e",
//    "c c=Save, k=v",
//    "c c=WriteWeights, k=z",
//    "c c=DisplayWeights, k=y",
//    "c c=InitData, k=n",
//    "c c=BoundingBoxes, k=j",
    "n n=x, t=i, i=1, d=0",
    "n n=epochs, t=i, i=2, d=100000",
    "n n=learn, t=r, i=1, d=0.01",
    "n n=color, t=i, i=3, d=0",
    "n n=displayFreq, t=i, i=4, d=1",
    "n n=net, t=i, i=5, d=3",
    "n n=type, t=i, i=6, d=1",
    "n n=rows, t=i, i=7, d=0",
    "n n=k, t=i, i=8, d=5",
    "n n=pane, t=i, i=9, d=3",
    "n n=ratioD, t=r, i=2, d=0.0",
    "n n=it, t=i, i=11, d=100",
    "n n=bias, t=r, i=3, d=0.0",
    "n n=y, t=i, i=12, d=0",
    "n n=true, t=i, i=13, d=5",
    "n n=predict, t=i, i=14, d=3",
    "n n=norm, t=i, i=15, d=2",
    "n n=count, t=i, i=16, d=54",
    "n n=L0, t=s, i=1, d=0",
    "n n=L1, t=s, i=2, d=0",
    "n n=L2, t=s, i=3, d=0",
    "n n=L3, t=s, i=4, d=0",
    "n n=L4, t=s, i=5, d=0",
    "n n=L5, t=s, i=6, d=0",
    "n n=L6, t=s, i=7, d=2",
    "n n=L7, t=s, i=8, d=20",
    "n n=L8, t=s, i=9, d=20",
    "n n=L9, t=s, i=10, d=6",
    "n n=x2, t=i, i=17, d=43000",
    "n n=mode, t=i, i=18, d=4",
    "n n=validRatio, t=r, i=4, d=0.25",
    "n n=trainSet, t=i, i=19, d=1",
    "n n=NN, t=i, i=20, d=1",
    "n n=big, t=i, i=21, d=1",
    "n n=maxY, t=r, i=5, d=1.0",
    "n n=minY, t=r, i=6, d=0.9",
    "n n=displayFreq2, t=i, i=22, d=100",
    "n n=angle, t=i, i=23, d=13",
    "n n=layer, t=i, i=26, d=0",
    "n n=channel, t=i, i=27, d=0",
    "n n=z, t=i, i=28, d=1",
    "n n=removeHeader, t=i, i=29, d=1",
    "n n=dataFile, t=f, i=11, d=train.csv",
    "n n=decay, t=r, i=7, d=0.95",
    "n n=divideBy, t=r, i=8, d=255.0",
    "n n=subtractBy, t=r, i=9, d=0.0",
    "n n=classify, t=i, i=30, d=0",
    "n n=scaleWeights, t=r, i=10, d=1.414",
    "n n=scale, t=r, i=11, d=0.13",
    "n n=conv, t=i, i=31, d=0",
    "n n=pool, t=i, i=32, d=1",
    "n n=dense, t=i, i=33, d=1",
    "n n=ratioA, t=r, i=12, d=1.0",
    "n n=col1Name, t=s, i=12, d=ImageId",
    "n n=col2Name, t=s, i=13, d=Label",
    "n n=row1Num, t=i, i=34, d=1",
    "n n=fileName, t=s, i=14, d=submit.csv",
    "n n=augment, t=i, i=35, d=0",
    "n n=removeCol1, t=i, i=36, d=0",
    "n n=xshift, t=r, i=13, d=2.8",
    "n n=yshift, t=r, i=14, d=2.8",
    "r c=InitData, n=z",
    "r c=NNpredict, n=trainSet",
    "r c=NNpredict, n=x",
    "r c=Dream, n=x",
    "r c=Dream, n=y",
    "r c=Dream, n=it",
    "r c=Dream, n=learn",
    "r c=Dream, n=bias",
    "r c=Dream, n=displayFreq",
    "r c=Train-Net, n=epochs",
    "r c=Train-Net, n=learn",
    "r c=Train-Net, n=decay",
    "r c=Train-Net, n=displayFreq",
    "r c=Train-Net, n=maxY",
    "r c=Train-Net, n=minY",
    "r c=Train-Net, n=mode",
    "r c=DotColor, n=color",
    "r c=Init-Net, n=net",
    "r c=Init-Net, n=scaleWeights",
    "r c=Activation, n=type",
    "r c=Validate-kNN, n=big",
    "r c=Validate-kNN, n=k",
    "r c=Validate-kNN, n=norm",
    "r c=Validate-kNN, n=displayFreq",
    "r c=DropOut, n=ratioD",
    "r c=DropOut, n=conv",
    "r c=DropOut, n=pool",
    "r c=DropOut, n=dense",
    "r c=Display, n=trainSet",
    "r c=Display, n=x",
    "r c=Display, n=count",
    "r c=Display, n=pane",
    "r c=Display, n=big",
    "r c=Display, n=layer",
    "r c=Display, n=channel",
    "r c=Display, n=classify",
    "r c=Display, n=NN",
    "r c=Display, n=augment",
    "r c=Init-Net, n=L0",
    "r c=Init-Net, n=L1",
    "r c=Init-Net, n=L2",
    "r c=Init-Net, n=L3",
    "r c=Init-Net, n=L4",
    "r c=Init-Net, n=L5",
    "r c=Init-Net, n=L6",
    "r c=Init-Net, n=L7",
    "r c=Init-Net, n=L8",
    "r c=Init-Net, n=L9",
    "r c=Save, n=x",
    "r c=Save, n=x2",
    "r c=Find-kNN, n=trainSet",
    "r c=Find-kNN, n=big",
    "r c=Find-kNN, n=x",
    "r c=Find-kNN, n=k",
    "r c=Find-kNN, n=norm",
    "r c=Find-kNN, n=pane",
    "r c=Load, n=trainSet",
    "r c=Load, n=rows",
    "r c=Load, n=validRatio",
    "r c=Load, n=removeHeader",
    "r c=Load, n=removeCol1",
    "r c=Load, n=dataFile",
    "r c=Load, n=divideBy",
    "r c=Load, n=subtractBy",
    "r c=Predict, n=NN",
    "r c=Predict, n=big",
    "r c=Predict, n=k",
    "r c=Predict, n=norm",
    "r c=Predict, n=displayFreq2",
    "r c=Predict, n=col1Name",
    "r c=Predict, n=col2Name",
    "r c=Predict, n=row1Num",
    "r c=Predict, n=fileName",
    "r c=DisplayAugment, n=x",
    "r c=DisplayAugment, n=count",
    "r c=DisplayAugment, n=ratioA",
    "r c=DisplayAugment, n=angle",
    "r c=DisplayAugment, n=scale",
    "r c=DisplayAugment, n=xshift",
    "r c=DisplayAugment, n=yshift",
    "r c=DisplayAugment, n=pane",
    "r c=DisplayAugment, n=big",
    "r c=DataAugment, n=ratioA",
    "r c=DataAugment, n=angle",
    "r c=DataAugment, n=scale",
    "r c=DataAugment, n=xshift",
    "r c=DataAugment, n=yshift",
    "s n=color, v=0, l=red",
    "s n=color, v=1, l=orange",
    "s n=color, v=2, l=yellow",
    "s n=color, v=3, l=green",
    "s n=color, v=4, l=blue",
    "s n=color, v=5, l=purple",
    "s n=net, v=0, l=custom",
// NAMES OF PRE-DEFINED NETS
    "s n=net, v=1, l=2-20-20-6",
    "s n=net, v=2, l=LeNet-5",
    "s n=net, v=3, l=784-C5:12-P2-C5:24-P2-128-10",
    "s n=net, v=4, l=196-100-10",
    "s n=net, v=5, l=784-C5:6-P2-50-10",
    //"s n=net, v=6, l=16-C3:2-P2-2-2",
// NAMES ABOVE
    "s n=type, v=1, l=ReLU",
    "s n=type, v=2, l=TanH",
    "s n=mode, v=2, l=2fps",
    "s n=mode, v=4, l=10fps",
    "s n=mode, v=5, l=20fps",
    "s n=mode, v=6, l=30fps",
    "s n=trainSet, v=1, l=train_set",
    "s n=trainSet, v=0, l=test_set",
    "s n=NN, v=0, l=usekNN",
    "s n=NN, v=1, l=useNN",
    "s n=big, v=0, l=use196",
    "s n=big, v=1, l=use784",
    "s n=big, v=2, l=other",
    "s n=removeHeader, v=0, l=no",
    "s n=removeHeader, v=1, l=yes",
    "s n=removeCol1, v=0, l=no",
    "s n=removeCol1, v=1, l=yes",
    "s n=classify, v=0, l=no",
    "s n=classify, v=1, l=yes",
    "s n=augment, v=0, l=no",
    "s n=augment, v=1, l=yes",
    "s n=divideBy, v=255.0, l=pixel_data",
    "s n=divideBy, v=1.0, l=no_scaling",
    "s n=subtractBy, v=0.0, l=no_bias"
};
// NUMBER BITMAPS
int digits[10][15] =
       {{1,1,1, 1,0,1, 1,0,1, 1,0,1, 1,1,1}, //0
        {0,1,0, 0,1,0, 0,1,0, 0,1,0, 0,1,0}, //1
        {1,1,1, 0,0,1, 1,1,1, 1,0,0, 1,1,1}, //2
        {1,1,1, 0,0,1, 1,1,1, 0,0,1, 1,1,1}, //3
        {1,0,1, 1,0,1, 1,1,1, 0,0,1, 0,0,1}, //4
        {1,1,1, 1,0,0, 1,1,1, 0,0,1, 1,1,1}, //5
        {1,1,1, 1,0,0, 1,1,1, 1,0,1, 1,1,1}, //6
        {1,1,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1}, //7
        {1,1,1, 1,0,1, 1,1,1, 1,0,1, 1,1,1}, //8
        {1,1,1, 1,0,1, 1,1,1, 0,0,1, 0,0,1}}; //9
// LETTER BITMAPS
int letters[26][15] =
       {{0,1,0, 1,0,1, 1,1,1, 1,0,1, 1,0,1}, //A
        {1,1,0, 1,0,1, 1,1,0, 1,0,1, 1,1,0}, //B
        {1,1,1, 1,0,0, 1,0,0, 1,0,0, 1,1,1}, //C
        {1,1,0, 1,0,1, 1,0,1, 1,0,1, 1,1,0}, //D
        {1,1,1, 1,0,0, 1,1,1, 1,0,0, 1,1,1}, //E
        {1,1,1, 1,0,0, 1,1,0, 1,0,0, 1,0,0}, //F
        {1,1,1, 1,0,0, 1,0,1, 1,0,1, 1,1,1}, //G
        {1,0,0, 1,0,1, 1,1,1, 1,0,1, 1,0,1}, //H
        {1,1,1, 0,1,0, 0,1,0, 0,1,0, 1,1,1}, //I
        {1,1,1, 0,1,0, 0,0,1, 1,0,1, 1,1,1}, //J
        {1,0,1, 1,0,1, 1,1,0, 1,0,1, 1,0,1}, //K
        {1,0,0, 1,0,0, 1,0,0, 1,0,0, 1,1,1}, //L
        {1,0,1, 1,1,1, 1,0,1, 1,0,1, 1,0,1}, //M
        {1,0,1, 1,1,1, 1,1,1, 1,1,1, 1,0,1}, //N
        {0,1,0, 1,0,1, 1,0,1, 1,0,1, 0,1,0}, //O
        {1,1,0, 1,0,1, 1,1,0, 1,0,0, 1,0,0}, //P
        {1,1,1, 1,0,1, 1,0,1, 1,1,1, 0,0,1}, //Q
        {1,1,1, 1,0,1, 1,1,1, 1,1,0, 1,0,1}, //R
        {1,1,1, 1,0,0, 1,1,1, 0,0,1, 1,1,1}, //S
        {1,1,1, 0,1,0, 0,1,0, 0,1,0, 0,1,0}, //T
        {1,0,1, 1,0,1, 1,0,1, 1,0,1, 1,1,1}, //U
        {1,0,1, 1,0,1, 1,0,1, 1,0,1, 0,1,0}, //V
        {1,0,1, 1,0,1, 1,0,1, 1,1,1, 1,0,1}, //W
        {1,0,1, 1,1,1, 1,1,1, 1,1,1, 1,0,1}, //X
        {1,0,1, 1,0,1, 0,1,0, 0,1,0, 0,1,0}, //Y
        {1,1,1, 0,0,1, 0,1,0, 1,0,0, 1,1,1}}; //Z

// TRAINING AND VALIDATION DATA
char* data = 0;
float (*trainImages)[784] = 0;
float (*trainImages2)[196] = 0;
int *trainDigits = 0;
int trainSizeI = 0, extraTrainSizeI = 1000;
int trainColumns = 0, trainSizeE = 0;
int *trainSet = 0; int trainSetSize = 0;
int *validSet = 0; int validSetSize = 0;
float *ents = 0, *ents2 = 0;
float *accs = 0, *accs2 = 0;
// TEST DATA
float (*testImages)[784] = 0;
float (*testImages2)[196] = 0;
int *testDigits;
int testSizeI = 0;
int testColumns = 0;
// MISC
float scaleMin = 0.9, scaleMax = 1.0;
char *weightsFile1 = "weights1.txt";
char *weightsFile2 = "weights2.txt";
// DOT DATA
int maxDots=250;
float trainDots[250][2];
int trainColors[250];
int trainSizeD = 0;
// NETWORK VARIABLES
int inited = -1;
int activation = 1; //1=ReLU, 2=TanH
int randomizeDescent = 1;
float an = 0.01;
int DOconv=1, DOdense=1, DOpool=1;
float dropOutRatio = 0.0, decay = 1.0;
float augmentRatio = 0.0, weightScale = 1.0;
float augmentScale = 0, imgBias=0.0;
int augmentAngle = 0;
float augmentDx = 0.0, augmentDy = 0.0;
float* layers[10] = {0};
int* dropOut[10] = {0};
float*  weights[10] = {0};
float* errors[10] = {0};
char layerNames[10][20] = {0};
int layerSizes[10] = {0};
int layerConv[10] = {0};
int layerPad[10] = {0};
int layerWidth[10] = {0};
int layerChan[10] = {0};
int layerStride[10] = {0};
int layerConvStep[10] = {0};
int layerConvStep2[10] = {0};
int layerType[10] = {0}; //0FC, 1C, 2P
int numLayers = 0;
// PREDEFINED ARCHITECTURES
char nets[6][10][20] =
          {{"","","","","","","","","",""},
           {"","","","","","","2","20","20","6"},
           {"","","","784","C5:6","P2","C5:16","P2","128","10"},
           {"","","","784","C5:12","P2","C5:24","P2","128","10"},
           {"","","","","","","","196","100","10"},
           {"","","","","","784","C5:6","P2","50","10"}};
           //{"","","","","","16","C3:2","P2","2","2"}};
// RAINBOW COLORS
double red[8] =   {1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0};
double green[8] = {0.0, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0};
double blue[8] =  {0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0};
double red2[256],green2[256],blue2[256];
int image[400][600] = {{0}};
int image2[80][120] = {{0}};
// DOT VARIABLES
int useSmall = -1;
int removeMode = -1;
int dotsMode = 4; //6=fluid display 2=slower
// THREAD VARIABLES
pthread_t workerThread;
pthread_attr_t 	stackSizeAttribute;
int pass[5] = {0};
int working = 0;
int requiredStackSize = 8*1024*1024;
// CONFUSION MATRIX DATA
int maxCD = 54;
int cDigits[10][10][54];
int showAcc = 1;
int showEnt = 1;
int showCon = 0;
int showDig[3][55] = {{0}};

/**********************************************************************/
/*      MAIN ROUTINE                                                  */
/**********************************************************************/
int main(int argc, char *argv[]){
    if (argc>1) printf("Ignoring unknown argument(s)\n");
    srand(time(0));
    int i, offset=0;
    char cmd, str[80], buffer[80];
    
    // INITIALIZE WEBGUI
    initParameterMap((char*)init,lines);
    webinit((char*)init,lines);
    websetmode(2);
    websettitle("MNIST");
    while (webstart(15000+offset)<0) offset++;
    websetcolors(8,red,green,blue,3);
    for (i=0;i<240000;i++) ((int*)image)[i]=6;
    webimagedisplay(600,400,(int*)image,3);
    for (i=0;i<256;i++){
        red2[i] = (double)i/255.0;
        green2[i] = (double)i/255.0;
        blue2[i] = (double)i/255.0;
    }
    pthread_attr_init (&stackSizeAttribute);
    pthread_attr_setstacksize (&stackSizeAttribute, requiredStackSize);
    
    // MAIN LOOP TO PROCESS COMMANDS
    while (1){
    
        webreadline(str);
        cmd = processCommand(str);
        
        if (cmd=='l'){ // LOAD
            int ct = ipGet("rows");
            double v = rpGet("validRatio");
            int t = ipGet("trainSet");
            int sh = ipGet("removeHeader");
            int rc = ipGet("removeCol1");
            float imgScale = rpGet("divideBy");
            imgBias = rpGet("subtractBy");
            if (t==1){
                webwriteline("Loading training images, please wait...");
                int x = loadTrain(ct,v,sh,imgScale,imgBias);
                sprintf(buffer,"Loaded %d rows training, %d features, vSetSize=%d",x,trainColumns,validSetSize);
                webwriteline(buffer);
            }
            else{
                webwriteline("Loading test images, please wait...");
                int x = loadTest(ct,sh,rc,imgScale,imgBias);
                sprintf(buffer,"Loaded %d rows test, %d features",x,testColumns);
                webwriteline(buffer);
            }
        }
        else if (cmd=='p'){ // DISPLAY
            int x = ipGet("x");
            int p = ipGet("pane");
            int ct = ipGet("count");
            int t = ipGet("trainSet");
            int lay = ipGet("layer");
            int chan = ipGet("channel");
            int cfy = ipGet("classify");
            int big = ipGet("big");
            int aug = ipGet("augment");
            //int nn = ipGet("NN");
            char nm[10] = "training";
            char ext[20] = " with label";
            if (t==0) {
                strcpy(nm,"test");
                strcpy(ext,"");
            }
            if (cfy==1) strcpy(ext," with prediction");
            if (lay!=0){
                if (aug==1) lay = -lay;
                displayDigit(x,ct,p,lay,chan,t,cfy,big);
                if (lay!=0) sprintf(buffer,"Displaying layer %d, channel %d activations, x=%d",abs(lay),chan,x);
                webwriteline(buffer);
            }
            else if (aug==1){
                float rat = 1.0;
                int p = ipGet("pane");
                int r = ipGet("angle");
                float sc = rpGet("scale");
                float dx = rpGet("xshift");
                float dy = rpGet("yshift");
                viewAugment(x,ct,rat,r,sc,dx,dy,p,big,t,cfy);
            }
            else {
                displayDigit(x,ct,p,0,0,t,cfy,big);
                if (ct>1) sprintf(buffer,"Displaying %s images %d to %d%s",nm,x,x+ct-1,ext);
                else sprintf(buffer,"Displaying %s image %d%s",nm,x,ext);
                webwriteline(buffer);
            }
        }
        else if (cmd=='i'){ // INIT-NET
            int t = ipGet("net");
            weightScale = rpGet("scaleWeights");
            initNet(t);
            sprintf(buffer,"Initialized NN=%d with Xavier init scaled=%.3f",t,weightScale);
            webwriteline(buffer);
            int len = sprintf(buffer,"Architecture (%s",layerNames[0]);
            for (i=1;i<10;i++) len += sprintf(buffer+len,"-%s",layerNames[i]);
            sprintf(buffer+len,")");
            webwriteline(buffer);
        }
        else if (cmd=='a'){ // ACTIVATION
            int a = ipGet("type");
            if (a<1 || a>2) webwriteline("Invalid activation");
            else {
                activation = a;
                sprintf(buffer,"Activation=%d where 1=ReLU, 2=TanH",a);
                webwriteline(buffer);
                if (a==2) webbutton(1,"Activation");
                else webbutton(-1,"Activation");
            }
        }
        else if (cmd=='o'){ // DROPOUT
            dropOutRatio = rpGet("ratioD");
            DOconv = ipGet("conv");
            DOpool = ipGet("pool");
            DOdense = ipGet("dense");
            sprintf(buffer,"DropOutRatio = %f, conv=%d, pool=%d, dense=%d",dropOutRatio,DOconv,DOpool,DOdense);
            webwriteline(buffer);
            if (dropOutRatio>0.0) webbutton(1,"DropOut");
            else webbutton(-1,"DropOut");
        }
        else if (cmd=='d'){ // DATA AUGMENTATION
            augmentRatio = rpGet("ratioA");
            augmentAngle = abs(ipGet("angle"));
            augmentScale = fabs(rpGet("scale"));
            augmentDx = rpGet("xshift");
            augmentDy = rpGet("yshift");
            sprintf(buffer,"AugmentRatio = %f, Angle = %d, Scale = %.2f, Dx = %.1f, Dy = %.1f",
                augmentRatio,augmentAngle,augmentScale,augmentDx,augmentDy);
            webwriteline(buffer);
            if (augmentRatio>0.0) webbutton(1,"DataAugment");
            else webbutton(-1,"DataAugment");
        }
        else if (cmd=='b'){ // TRAIN-NET
            an = rpGet("learn");
            scaleMin = rpGet("minY");
            scaleMax = rpGet("maxY");
            decay = rpGet("decay");
            if (working==1){
                sprintf(buffer,"wait until learning ends, learn=%f",an);
                webwriteline(buffer);
            }
            else{
                int x = ipGet("epochs");
                int y = ipGet("displayFreq");
                dotsMode = ipGet("mode");
                sprintf(buffer,"Beginning %d epochs with lr=%f and decay=%f",x,an,decay);
                webwriteline(buffer);
                pass[0]=x; pass[1]=y; pass[2]=1; working=1;
                pthread_create(&workerThread,&stackSizeAttribute,runBackProp,NULL);
            }
        }
        else if (cmd=='g'){ // FIND-KNN
            int t = ipGet("trainSet");
            int big = ipGet("big");
            int x = ipGet("x");
            int k = ipGet("k");
            int d = ipGet("norm");
            int p = ipGet("pane");
            int c = singleKNN(x,k,d,p,t,big,1);
            sprintf(buffer,"kNN predicts %d for image %d with k=%d and L%d norm",c,x,k,d);
            webwriteline(buffer);
        }
        else if (cmd=='k'){ // VALIDATE-KNN
            if (working==1){
                webwriteline("wait until learning ends");
            }
            else{
                int x = ipGet("k");
                int y = ipGet("displayFreq");
                int big = ipGet("big");
                int d = ipGet("norm");
                pass[0]=big; pass[1]=y; pass[2]=x; pass[3]=d;
                pass[4]=1; working=1;
                pthread_create(&workerThread,&stackSizeAttribute,runKNN,NULL);
            }
        }
        else if (cmd=='t'){ // STOP
            working=0;
        }
        else if (cmd=='w'){ // PREDICT
            int NN = ipGet("NN");
            int k = ipGet("k");
            int d = ipGet("norm");
            int y = ipGet("displayFreq2");
            int big = ipGet("big");
            writePredictFile(NN,k,d,y,big);
        }
        else if (cmd=='q'){ // QUIT
            if (working==1) pthread_cancel(workerThread);
            webwriteline("QUITTING. Good-bye");
            usleep(500000);
            webstop();
            return 0;
        }
        
        else if (cmd=='c'){ // CLEAR PANE 3
            trainSizeD = 0;
            updateImage();
            webwriteline("Pane 3 cleared. Dots cleared");
        }
        else if (cmd=='r'){ // DOT REMOVE MODE
            if (removeMode==-1) removeMode = 1;
            else removeMode = -1;
            webbutton(removeMode,"DotRemoveMode");
            sprintf(buffer,"Remove mode = %d",removeMode);
            webwriteline(buffer);
        }
        else if (cmd=='u'){ // DOT LOW RES MODE
            if (useSmall==-1) useSmall = 1;
            else useSmall = -1;
            webbutton(useSmall,"DotLowResMode");
            sprintf(buffer,"UseSmall = %d",useSmall);
            webwriteline(buffer);
        }
        
        // SPECIAL FEATURES, UNCOMMENT IN MENU TO TURN ON
        
        else if (cmd=='j') // BOUNDING BOXES
            boundingBoxes();
        else if (cmd=='z'){ // WRITE WEIGHTS
            writeFile2();
        }
        else if (cmd=='y'){ // DISPLAY WEIGHTS
            webwriteline("Weights displayed in shell");
            displayWeights();
        }
        else if (cmd=='n'){ // INIT DATA
            int z = ipGet("z");
            sprintf(buffer,"Initing data with z=%d",z);
            webwriteline(buffer);
            initData(z);
        }
        else if (cmd=='v'){ // SAVE
            int x = ipGet("x");
            int x2 = ipGet("x2");
            for (i=0;i<196;i++) trainImages2[x2][i] = trainImages2[x][i];
            for (i=0;i<784;i++) trainImages[x2][i] = trainImages[x][i];
            trainDigits[x2] = trainDigits[x];
            sprintf(buffer,"Saved image %d to %d",x,x2);
            webwriteline(buffer); 
        }
        else if (cmd=='e'){ // DREAM
            int x = ipGet("x");
            int y = ipGet("y");
            int it = ipGet("it");
            an = rpGet("learn");
            int ds = ipGet("displayFreq");
            float bs = rpGet("bias");
            sprintf(buffer,"Dreaming x=%d to y=%d, it=%d, an=%f",x,y,it,an);
            webwriteline(buffer);
            dream(x,y,it,bs,ds);
        }
        else if (cmd=='m'){ // PROCESS MOUSE CLICK
            float x = atof(extractVal(str,'x'));
            float y = atof(extractVal(str,'y'));
            int b = atoi(extractVal(str,'n'));
            int p = atoi(extractVal(str,'e'));
            int r, row, col, d, num, dd;
            float c;
            if (showDig[p-3][0]!=0){
                d = showDig[p-3][0];
                if (d<=6) {r=2; c=3;}
                else if (d<=12) {r=3; c=4.5;}
                else if (d<=24) {r=4; c=6;}
                else if (d<=35) {r=5; c=7;}
                else if (d<=54) {r=6; c=9;}
                col = (int)(x*c/1.5);
                row = (int)((1-y)*r);
                if (col<0) col=0; if (col>=(int)c) col=c-1;
                if (row<0) row=0; if (row>=r) row=r-1;
                dd = 1+row*c+col;
                num = showDig[p-3][dd];
              
                if (x>1.48 && (1-y)<0.025){
                   if (p==3) showCon = 1;
                   else if (p==4) showEnt = 1;
                   else if (p==5) showAcc = 1;
                   clearImage(p);
                }
                else if (num!=-1 && dd<=showDig[p-3][0]){
                    sprintf(buffer,"Clicked image %d, digit %d",num,trainDigits[num]);
                    webwriteline(buffer);
                    ipSet("x",num);
                    webupdate(ip,rp,sp);
                }
            }
            else if (showCon==0){
                int c = ipGet("color");
                if (p==3 && trainSizeD<100){
                    if (removeMode==1){
                        removeDot(x,y);
                    }
                    else{
                        trainDots[trainSizeD][0] = x;
                        trainDots[trainSizeD][1] = y;
                        trainColors[trainSizeD] = c + b;
                        trainSizeD++;
                    }
                    if (working==0) updateImage();
                }
            }
            else if (p==3){
                int col = (int)(x/0.1364)-1;
                int row = (int)((1-y)/0.0909)-1;
                if (col<0) col=0; if (col>9) col=9;
                if (row<0) row=0; if (row>9) row=9;
                displayCDigits(row,col);
            }
        }
    }
    return 0;
}

/**********************************************************************/
/*      SPECIAL AND/OR DEBUGGING                                      */
/**********************************************************************/
void boundingBoxes(){
    // CALCULATES VARIANCE OF BOUNDING BOXES
    webwriteline("Calculating bounding boxes");
    int i,j,k,found;
    int left[trainSizeI], right[trainSizeI];
    int top[trainSizeI], bottom[trainSizeI];
    
    for (k=0;k<trainSizeI;k++){
        top[k] = 0; bottom[k] = 28; left[k] = 0; right[k] = 28;
        for (i=0;i<28;i++){
            found = 0;
            for (j=0;j<28;j++)
                if (trainImages[k][28*i+j] != 0) found=1;
            if (found==1){
                top[k] = i;
                break;
            }
        }
        for (i=27;i>=0;i--){
            found = 0;
            for (j=0;j<28;j++)
                if (trainImages[k][28*i+j] != 0) found=1;
            if (found==1){
                bottom[k] = i;
                break;
            }
        }
        for (j=0;j<28;j++){
            found = 0;
            for (i=0;i<28;i++)
                if (trainImages[k][28*i+j] != 0) found=1;
            if (found==1){
                left[k] = j;
                break;
            }
        }
        for (j=27;j>=0;j--){
            found = 0;
            for (i=0;i<28;i++)
                if (trainImages[k][28*i+j] != 0) found=1;
            if (found==1){
                right[k] = j;
                break;
            }
        }
    }
    int sum1 = 0, sum2 = 0;
    int wd1[10]={0}, wd2[10]={0}, ct[10]={0};
    float wd1avg[10], wd2avg[10], xavg, yavg;
    for (k=0;k<trainSizeI;k++){
        wd1[trainDigits[k]] += right[k] - left[k] + 1;
        wd2[trainDigits[k]] += bottom[k] - top[k] + 1;
        ct[trainDigits[k]] ++;
        sum1 += right[k] + left[k];
        sum2 += bottom[k] + top[k];
    }
    char buffer[80];
    
    float wXavg = 0.0, wYavg = 0.0;
    for (k=0;k<10;k++){
        wd1avg[k] = (float)wd1[k]/ct[k];
        wd2avg[k] = (float)wd2[k]/ct[k];
        wXavg += (float)wd1[k];
        wYavg += (float)wd2[k];
    }
    wXavg = wXavg/trainSizeI;
    wYavg = wYavg/trainSizeI;
    
    xavg = sum1/2.0/trainSizeI;
    yavg = sum2/2.0/trainSizeI;
    float varScaleX = 0.0, varScaleY = 0.0;
    float varShiftX = 0.0, varShiftY = 0.0;
    
    for (k=0;k<trainSizeI;k++){
        varScaleX += square( (right[k] - left[k] + 1)/wd1avg[trainDigits[k]] - 1.0);
        varScaleY += square( (bottom[k] - top[k] + 1)/wd2avg[trainDigits[k]] - 1.0);
        varShiftX += square( (right[k]+left[k])/2.0 - xavg );
        varShiftY += square( (bottom[k]+top[k])/2.0 - yavg );
    }
    
    varScaleX = sqrt (varScaleX / trainSizeI);
    varScaleY = sqrt (varScaleY / trainSizeI);
    varShiftX = sqrt (varShiftX / trainSizeI);
    varShiftY = sqrt (varShiftY / trainSizeI);
    
    sprintf(buffer,"Avg center=%.1f,%.1f has sdX=%.3f sdY=%.3f",xavg,yavg,varShiftX,varShiftY);
    webwriteline(buffer);
    sprintf(buffer,"Avg widthX=%.1f avg widthY=%.1f has sdScaleX=%.3f, sdScaleY=%.3f",wXavg,wYavg,varScaleX,varScaleY);
    webwriteline(buffer);
}

float square(float x){
    return x*x;
}

/**********************************************************************/
/*      SPECIAL AND/OR DEBUGGING                                      */
/**********************************************************************/
void initData(int z){
    // DEBUG FEATURE MAKES 4X4 IMAGE
    trainSizeI = 1;
    trainSetSize = 1;
    trainSet[0] = 0;
    for (int i=0;i<16;i++) trainImages[0][i]=1.0;
    trainImages[0][5] = 0.0;
    trainImages[0][6] = 0.0;
    trainImages[0][9] = 0.0;
    trainImages[0][10] = 0.0;
    trainColumns = 16;
    trainDigits[0] = z;
}

/**********************************************************************/
/*      SPECIAL AND/OR DEBUGGING                                      */
/**********************************************************************/
void displayWeights(){
    // DEBUG FEATURE DISPLAYS WEIGHTS, ACTIVATIONS, ERRORS, DROPOUT
    int i,j,ws;
    char name[10];
    printf("Weights:\n");
    for (i=11-numLayers;i<10;i++){
        if (layerType[i]==0){ // FULLY CONNECTED
            ws = layerSizes[i] * (layerSizes[i-1]*layerChan[i-1]+1);
            strcpy(name,"FC");
        }
        else if (layerType[i]==1){ // CONVOLUTION
            ws = (layerConvStep[i]+1) * layerChan[i];
            strcpy(name,"conv");
        }
        else if (layerType[i]==2){ // POOLING
            ws = 1;
            strcpy(name,"pool");
        }
        printf("Layer %d (%s):\n",i,name);
        for (j=0;j<ws;j++){
            printf(", %.3f",weights[i][j]);
        }
        printf("\n");
    }
    printf("Activations:\n");
    for (i=10-numLayers;i<10;i++){
        printf("LAYER %d: ",i);
        for (j=0;j<layerSizes[i]*layerChan[i]+1;j++){
            printf(", %.3f",layers[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("Dropout:\n");
    for (i=11-numLayers;i<9;i++){
        printf("LAYER %d: ",i);
        for (j=0;j<layerSizes[i]*layerChan[i];j++){
            printf(", %d",dropOut[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("Errors:\n");
    for (i=11-numLayers;i<10;i++){
        printf("LAYER %d: ",i);
        for (j=0;j<layerSizes[i]*layerChan[i];j++){
            printf(", %.3e",errors[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**********************************************************************/
/*      DISPLAY DOTS                                                  */
/**********************************************************************/
void removeDot(float x, float y){
    // REMOVES DOT THAT USER UNCLICKED
    float d, min = 10;
    int i,imin = -1;
    for (i=0;i<trainSizeD;i++){
        d = pow(x-trainDots[i][0],2) + pow(y-trainDots[i][1],2);
        if (d<min){
            min = d;
            imin = i;
        } 
    }
    if (imin!=-1){
        trainDots[imin][0] = trainDots[trainSizeD-1][0];
        trainDots[imin][1] = trainDots[trainSizeD-1][1];
        trainColors[imin] = trainColors[trainSizeD-1];
        trainSizeD--;
    }
}

/**********************************************************************/
/*      DISPLAY DOTS                                                  */
/**********************************************************************/
void placeDots(){
    // DRAWS DOTS TO PANE 3
    int i,j,k,x,y;
    for (i=0;i<trainSizeD;i++){
        x = trainDots[i][0]*400 - 7;
        y = trainDots[i][1]*400 - 7;
        for (j=0;j<15;j++)
        for (k=0;k<15;k++){
            if (y+j>=0 && y+j<400 && x+k>=0 && x+k<600){
                image[y+j][x+k] = trainColors[i];
                if (j<2 | j>12 | k<2 | k>12)
                image[y+j][x+k] = 7;
            }
        }
    }
}

/**********************************************************************/
/*      DISPLAY DOTS                                                  */
/**********************************************************************/
void updateImage(){
    // UPDATES PANE 3 IMAGE WITH DOTS
    int i,j;
    websetcolors(8,red,green,blue,3);
    for (i=0;i<240000;i++) ((int*)image)[i]=6;
    placeDots();
    if (useSmall==-1) webimagedisplay(600,400,(int*)image,3);
    else{
        for (i=0;i<80;i++)
        for (j=0;j<120;j++)
            image2[i][j] = image[i*5][j*5];
        webimagedisplay(120,80,(int*)image2,3);
    }
    showDig[0][0]=0;
    showCon=0;
}

/**********************************************************************/
/*      DISPLAY DOTS                                                  */
/**********************************************************************/
void clearImage(int p){
    // CLEARS IMAGE IN PANE P
    websetcolors(8,red,green,blue,p);
    for (int i=0;i<80;i++) for (int j=0;j<120;j++)
        image2[i][j] = 6;
    webimagedisplay(120,80,(int*)image2,p);
}

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
int isDigits(int init){
    // DETERMINES WHETHER TO TRAIN DOTS OR LOADED DATA
    int in = 10-numLayers;
    if (layerSizes[in]==196 || layerSizes[in]==784 || layerSizes[in]==trainColumns) return 1;
    else return 0;
}

/**********************************************************************/
/*      DISPLAY DATA                                                  */
/**********************************************************************/
void printDigit(int a,int b,int *canvas,int m,int n,float *digit,int row,int col,int d3,int d1,int d2){
    //DRAWS DIGIT INTO CANVAS STARTING AT (X,Y)=(B,A) FROM TOP LEFT CORNER
    //D3 IS STRIDE, D1 IS PIXEL WIDTH, D2 IS SPACING
    int dx = d1 + d2;
    int dy = d1 + d2;
    int i,j,ki,kj;
    for (i=0;i<row/d3;i++)
    for (j=0;j<col/d3;j++)
    for (ki=0;ki<d1;ki++)
    for (kj=0;kj<d1;kj++)
        if (m-1-(dx*i+ki+a)>=0 && m-1-(dx*i+ki+a)<m && dy*j+kj+b<n)
        canvas[n*(m-1-(dx*i+ki+a)) + dy*j+kj+b] = 255*digit[row*i*d3 + j*d3];
}

/**********************************************************************/
/*      DISPLAY DATA                                                  */
/**********************************************************************/
void printInt(int a,int b,int *canvas,int m,int n,int num,int col,int d1){
    // DRAWS INT ON CANVAS
    char str[80];
    sprintf(str,"%d",num);
    int i,j,k1,k2,w;
    for (w=0;w<strlen(str);w++)
    for (i=0;i<5;i++)
    for (j=0;j<3;j++)
    for (k1=0;k1<d1;k1++)
    for (k2=0;k2<d1;k2++)
        if (digits[str[w]-'0'][i*3+j] != 0)
        if (m-1-(a+(i*d1+k1))>=0 && m-1-(a+(i*d1+k1))<m && b+(j*d1+k2)+4*d1*w<n)
        canvas[n*(m-1-(a+(i*d1+k1))) + b+(j*d1+k2)+4*d1*w] = col;
}

/**********************************************************************/
/*      DISPLAY DATA                                                  */
/**********************************************************************/
void printStr(int a,int b,int *canvas,int m,int n,char *str,int col,int d1){
    // DRAWS STRING ON CANVAS
    // MAKE STR ALL LOWERCASE
    int i,j,k1,k2,w;
    for (w=0;w<strlen(str);w++)
    for (i=0;i<5;i++)
    for (j=0;j<3;j++)
    for (k1=0;k1<d1;k1++)
    for (k2=0;k2<d1;k2++)
        if (letters[str[w]-'a'][i*3+j] != 0)
        if (m-1-(a+(i*d1+k1))>=0 && m-1-(a+(i*d1+k1))<m && b+(j*d1+k2)+4*d1*w<n)
        canvas[n*(m-1-(a+(i*d1+k1))) + b+(j*d1+k2)+4*d1*w] = col;
}

/**********************************************************************/
/*      DISPLAY DATA                                                  */
/**********************************************************************/
void displayDigit(int x, int ct, int p, int lay, int chan, int train, int cfy, int big){
    // DISPLAYS 1 OR MORE IMAGES. OR DISPLAYS NET'S LAYER ACTIVATIONS
    if (p<3 || p>5) {
        webwriteline("Invalid pane");
        return;
    }
    float (*trainImagesB)[784] = trainImages;
    float (*trainImages2B)[196] = trainImages2;
    if (train==0){
        trainImagesB = testImages;
        trainImages2B = testImages2;
    }
    int i, j, k, imax, imin;
    // DISPLAY NET'S LAYER ACTIVATIONS
    if (lay!=0){
        if (lay<0){ // do data augment
            lay = -lay;
            doAugment(x,big,train);
            x = trainSizeE;
            train = 1;
        }
        forwardProp(x,0,train);
        for (k=0;k<ct;k++){
            float min=1e6, max=-1e6;
            for (i=0;i<layerSizes[lay];i++) {
                trainImages[trainSizeE+k][i] = layers[lay][(chan+k)*layerSizes[lay] + i];
                if (trainImages[trainSizeE+k][i]>max) {
                    max = trainImages[trainSizeE+k][i];
                    imax = i;
                }
                if (trainImages[trainSizeE+k][i]<min) {
                    min = trainImages[trainSizeE+k][i];
                    imin = i;
                }
            }
            for (i=0;i<layerSizes[lay];i++) trainImages[trainSizeE+k][i] = (trainImages[trainSizeE+k][i] - min) / (max - min);
        }
        // DISPLAY SINGLE MAP'S ACTIVATIONS
        if (ct==1){
            websetcolors(256,red2,green2,blue2,p);
            for (i=0;i<240000;i++) ((int*)image)[i] = 128;
            printDigit(0,0,(int*)image,400,600,trainImages[trainSizeE],layerWidth[lay],layerWidth[lay],1,10,2);
            for (i=0;i<10;i++)
            for (j=0;j<10;j++){
                if (i==j || i+j==9) image[390+i][590+j] = 0;
                else image[390+i][590+j] = 255;
            }
            webimagedisplay(600,400,(int*)image,p);
            showDig[p-3][0]=1;
            if (p==4) showEnt = 0;
            else if (p==5) showAcc = 0;
            else if (p==3) showCon = 0;
        }
        // DISPLAY MULTIPLE MAP'S ACTIVATIONS
        else{
            int dgs[ct];
            for (int i=0;i<ct;i++) dgs[i] = trainSizeE+i;
            displayDigits(dgs,ct,p,1,0,layerWidth[lay],2);
        }
    }
    // DISPLAY IMAGES
    else{
        // DISPLAY 1 IMAGE
        if (ct==1){
            websetcolors(256,red2,green2,blue2,p);
            for (i=0;i<240000;i++) ((int*)image)[i] = 128;
            int wid;
            if (train==1) wid = (int)sqrt(trainColumns);
            else wid = (int)sqrt(testColumns);
            printDigit(0,15*12,(int*)image,400,600,trainImagesB[x],wid,wid,1,10,2);
            printDigit(0,0,(int*)image,400,600,trainImages2B[x],14,14,1,10,2);
            int dt = -1;
            if (train==1) dt = trainDigits[x];
            int nn = ipGet("NN");
            if (nn==1 && cfy==1 && layers[0]!=NULL) dt = forwardProp(x,0,train);
            else if (nn==0 && cfy==1)
                dt = singleKNN(x,ipGet("k"),ipGet("norm"),3,train,big,0);
            if (train!=0 || (cfy!=0 && nn==0) || (cfy!=0 && layers[0]!=NULL))
                printInt(14*12,14*12-7,(int*)image,400,600,dt,255,2);
            printInt(14*12,7,(int*)image,400,600,x,0,2);
            for (i=0;i<10;i++)
            for (j=0;j<10;j++){
                if (i==j || i+j==9) image[390+i][590+j] = 0;
                else image[390+i][590+j] = 255;
            }
            webimagedisplay(600,400,(int*)image,p);
            showDig[p-3][0]=1;
            if (p==4) showEnt = 0;
            else if (p==5) showAcc = 0;
            else if (p==3) showCon = 0;
        }
        // DISPLAY MULTIPLE IMAGES
        else{
            int dgs[ct];
            for (int i=0;i<ct;i++) dgs[i] = x+i;
            displayDigits(dgs,ct,p,train,cfy,0,big);
        }
    }
}

/**********************************************************************/
/*      DISPLAY DATA                                                  */
/**********************************************************************/
void doAugment(int img, int big, int t){
    // PERFORM DATA AUGMENTATION ON 1 IMAGE FOR LAYERS DISPLAY
    int i, j, rot=0, hres=0, lres=1;
    if (big==1) {hres=1; lres=0;}
    float xs, ys, sc2;
    int r = ipGet("angle");
    float sc = rpGet("scale");
    float dx = rpGet("xshift");
    float dy = rpGet("yshift");
    rot = (int)(2.0 * r * (float)rand()/(float)RAND_MAX - r);
    sc2 = 1.0 + 2.0 * sc * (float)rand()/(float)RAND_MAX - sc;
    xs = (2.0 * dx * (float)rand()/(float)RAND_MAX - dx);
    ys = (2.0 * dy * (float)rand()/(float)RAND_MAX - dy);
    dataAugment(img,rot,sc2,xs,ys,-1,hres,lres,t);
}

/**********************************************************************/
/*      DISPLAY DATA                                                  */
/**********************************************************************/
void viewAugment(int img, int ct, float ratio, int r, float sc, int dx, int dy, int p, int big, int t, int cfy){
    // DISPLAY IMAGES WITH DATA AUGMENTATION
    int i, j, rot=0, hres=0, lres=1;
    float xs, ys, sc2;
    if (big==2 &&layerSizes[10-numLayers]==784){hres=1;lres=0;}
    else if (big==1) {hres=1;lres=0;}
    for (i=ct-1;i>=0;i--){
        if ( (float)rand()/(float)RAND_MAX <= ratio ){
            rot = (int)(2.0 * r * (float)rand()/(float)RAND_MAX - r);
            sc2 = 1.0 + 2.0 * sc * (float)rand()/(float)RAND_MAX - sc;
            xs = (2.0 * dx * (float)rand()/(float)RAND_MAX - dx);
            ys = (2.0 * dy * (float)rand()/(float)RAND_MAX - dy);
        }
        else{ rot=0; sc2=1.0; xs=0.0; ys=0.0; }
        dataAugment(img+i,rot,sc2,xs,ys,-1,hres,lres,t);
        if (lres==1) for (j=0;j<196;j++) trainImages2[trainSizeE+i][j] = trainImages2[trainSizeE][j];
        if (hres==1) for (j=0;j<784;j++) trainImages[trainSizeE+i][j] = trainImages[trainSizeE][j];
        trainDigits[trainSizeE+i] = trainDigits[trainSizeE];
    }
    displayDigit(trainSizeE,ct,p,0,0,1,cfy,big);
}

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
void dataAugment(int img, int r, float sc, float dx, float dy, int p, int hiRes, int loRes, int t){
    // AUGMENT AN IMAGE AND STORE RESULT IN TRAIN IMAGES ARRAY
    int i,j,x2,y2;
    float x,y;
    float pi = 3.1415926;
    float c = cos(pi * r/180.0);
    float s = sin(pi * r/180.0);
    float (*trainImagesB)[784] = trainImages;
    float (*trainImages2B)[196] = trainImages2;
    if (t==0){
        trainImagesB = testImages;
        trainImages2B = testImages2;
    }
    if (loRes==1){
    for (i=0;i<14;i++)
    for (j=0;j<14;j++){
        x = (j - 6.5)/sc - dx/2.0;
        y = (i - 6.5)/sc + dy/2.0;
        x2 = (int)round((c*x-s*y)+6.5);
        y2 = (int)round((s*x+c*y)+6.5);
        if (y2>=0 && y2<14 && x2>=0 && x2<14)
            trainImages2[trainSizeE][i*14+j] = trainImages2B[img][y2*14+x2];
        else trainImages2[trainSizeE][i*14+j] = -imgBias;
    }}
    if (hiRes==1){
    for (i=0;i<28;i++)
    for (j=0;j<28;j++){
        x = (j - 13.5)/sc - dx;
        y = (i - 13.5)/sc + dy;
        x2 = (int)round((c*x-s*y)+13.5);
        y2 = (int)round((s*x+c*y)+13.5);
        if (y2>=0 && y2<28 && x2>=0 && x2<28)
            trainImages[trainSizeE][i*28+j] = trainImagesB[img][y2*28+x2];
        else trainImages[trainSizeE][i*28+j] = -imgBias;
    }}
    if (p>=3 && p<=5) displayDigit(trainSizeE,1,p,0,0,1,0,2);
    trainDigits[trainSizeE] = trainDigits[img];
    if (t==0) trainDigits[trainSizeE] = -1;
}

/**********************************************************************/
/*      DISPLAY PROGRESS                                              */
/**********************************************************************/
void line(int *img, int m, int n, float x1, float y1, float x2, float y2, int d, int c){
    // DRAW A LINE INTO CANVAS
    int xA, yA, xB, yB, xD, yD, d1, d2, s;
    float xDf = 1.0, yDf = 1.0;
    int i, j, k;
    int points[(int)(1.5*m*m)][2];
    int pointsSize = 0;
    xA = (int)(x1*m);
    yA = (int)(y1*m);
    xB = (int)(x2*m);
    yB = (int)(y2*m);
    s = abs(xB - xA);
    if (abs(yB - yA)>s) s = abs(yB - yA);
    if (s!=0){
        yDf = (float)(yB-yA)/(float)s;
        xDf = (float)(xB-xA)/(float)s;
    }
    for (i=0;i<s+1;i++){
        points[i][0] = (int)(xA+xDf*i);
        points[i][1] = (int)(yA+yDf*i);
    }
    pointsSize = s+1;
    for (i=0;i<pointsSize;i++)
    for (j=-d;j<1+d;j++)
    for (k=-d;k<1+d;k++){
        if (points[i][1]+j>=0 && points[i][1]+j<m && points[i][0]+k>=0 && points[i][0]+k<1.5*m)
        img[n*(points[i][1]+j) + points[i][0]+k] = c;
    }
}

/**********************************************************************/
/*      DISPLAY DOTS                                                  */
/**********************************************************************/
void displayClassify(int dd){
    // DISPLAY CLASSIFICATION REGIONS FOR DOTS
    int dx = 10, dy = 10, i,j,x,y,c;
    websetcolors(8,red,green,blue,3);
    for (i=0;i<240000;i++) ((int*)image)[i]=6;
    int xl = 600/dx - 1;
    int yl = 400/dy - 1;
    for (x=0; x<xl; x++)
    for (y=0; y<yl; y++){
        trainDots[maxDots-1][0] = (float)dx*((float)x+0.5)/400.0;
        trainDots[maxDots-1][1] = (float)dy*((float)y+0.5)/400.0;
        c = forwardProp(maxDots-1,0,1);
        for (i=x*dx;i<(x+1)*dx;i++)
        for (j=y*dy;j<(y+1)*dy;j++)
            image[j][i] = c;
    }
    placeDots();
    if (useSmall==-1 || dd==1) webimagedisplay(600,400,(int*)image,3);
    else{
        for (i=0;i<80;i++)
        for (j=0;j<120;j++)
            image2[i][j] = image[i*5][j*5];
        webimagedisplay(120,80,(int*)image2,3);
    }
    showDig[0][0]=0;
}

/**********************************************************************/
/*      DISPLAY PROGRESS                                              */
/**********************************************************************/
void displayEntropy(float *ents, int entSize, float *ents2, int display){
    // DISPLAY ENTROPY PLOT
    int i,j;
    int *img = (int*)image2;
    int w=120, h=80;
    if (useSmall==-1){
        img = (int*) image;
        w = 600;
        h = 400;
    }
    float x1,y1,x2,y2,y1B,y2B;
    for (i=0;i<w*h;i++) img[i]=6;
    line(img,h,w,0.1,0.9,0.1,0.1,0,7);
    line(img,h,w,0.1,0.1,1.4,0.1,0,7);
    int pwr = (int)log10f(entSize);
    int inc = (int)pow(10,pwr);
    for (i=1;i<entSize/inc+1;i++){
        if (useSmall==-1){ 
            line(img,h,w,0.1+1.3*(inc*i)/entSize,0.08,0.1+1.3*(inc*i)/entSize,0.12,0,7);
            printInt(370,(int)(40+520*(inc*i)/(float)entSize)-(pwr+1)*5,img,h,w,inc*i*display,0,3);
        }
        else{
            line(img,h,w,0.1+1.3*(inc*i)/entSize,0.1,0.1+1.3*(inc*i)/entSize,0.13,0,7);
            printInt(73,(int)(8+104*(inc*i)/(float)entSize)-(pwr+1)*1,img,h,w,inc*i*display,0,1);
        }
    }
    float dx = 1.3/(entSize-1);
    float ymax = 0;
    for (i=0;i<entSize;i++)
        if (ents[i]>ymax) ymax=ents[i];
    for (i=1;i<entSize;i++){
        x2 = 0.1 + dx*i;
        x1 = 0.1 + dx*(i-1);
        y2 = 0.1 + 0.8*(ents[i]/ymax);
        y1 = 0.1 + 0.8*(ents[i-1]/ymax);
        line(img,h,w,x1,y1,x2,y2,0,0);
        if (useSmall==-1)line(img,h,w,x1,y1,x1,y1,2,0);
        if (isDigits(inited)==1){
            y2B = 0.1 + 0.8*(ents2[i]/ymax);
            y1B = 0.1 + 0.8*(ents2[i-1]/ymax);
            line(img,h,w,x1,y1B,x2,y2B,0,4);
            if (useSmall==-1)line(img,h,w,x1,y1B,x1,y1B,2,4);
        }        
    }
    if (useSmall==-1) line(img,h,w,x2,y2,x2,y2,2,0);
    if (isDigits(inited)==1 && useSmall==-1) line(img,h,w,x2,y2B,x2,y2B,2,4);
    int d=3, ww=100, hh=10; if (useSmall==1) {d=1; ww=10; hh=1;}
    printStr(hh,ww,img,h,w,"entropy",0,d);
    websetcolors(8,red,green,blue,4);
    webimagedisplay(w,h,img,4);
    showDig[1][0]=0;
}

/**********************************************************************/
/*      DISPLAY PROGRESS                                              */
/**********************************************************************/
void displayAccuracy(float *accs, int accSize,float *accs2, int display){
    // DISPLAY ACCURACY PLOT
    int i,j;
    float min = 0.8, max = 1.0;
    min = scaleMin;
    max = scaleMax;
    int *img = (int*)image2;
    int w=120, h=80;
    if (useSmall==-1){
        img = (int*) image;
        w = 600;
        h = 400;
    }
    float x1,y1,x2,y2,y1B,y2B;
    for (i=0;i<w*h;i++) img[i]=6;
    line(img,h,w,0.1,0.9,0.1,0.1,0,7);
    line(img,h,w,0.1,0.9,1.4,0.9,0,7);
    line(img,h,w,0.1,0.1,1.4,0.1,0,7);
    for (i=0;i<11;i++){ 
        if (useSmall==-1) line(img,h,w,0.08,0.1+0.08*i,0.12,0.1+0.08*i,0,7);
        else line(img,h,w,0.1,0.1+0.08*i,0.12,0.1+0.08*i,0,7);
    }
    int pwr = (int)log10f(accSize);
    int inc = (int)pow(10,pwr);
    for (i=1;i<accSize/inc+1;i++){
        if (useSmall==-1){ 
            line(img,h,w,0.1+1.3*(inc*i)/accSize,0.08,0.1+1.3*(inc*i)/accSize,0.12,0,7);
            printInt(370,(int)(40+520*(inc*i)/(float)accSize)-(pwr+1)*5,img,h,w,inc*i*display,0,3);
        }
        else{
            line(img,h,w,0.1+1.3*(inc*i)/accSize,0.1,0.1+1.3*(inc*i)/accSize,0.13,0,7);
            printInt(73,(int)(8+104*(inc*i)/(float)accSize)-(pwr+1)*1,img,h,w,inc*i*display,0,1);
        }
    }
    //line(img,h,w,0.1,0.66,1.4,0.66,0,3);
    float dx = 1.3/(accSize-1);
    float ymax = max-min;
    for (i=1;i<accSize;i++){
        x2 = 0.1 + dx*i;
        x1 = 0.1 + dx*(i-1);
        y2 = 0.1 + 0.8*((accs[i]-min)/ymax);
        y1 = 0.1 + 0.8*((accs[i-1]-min)/ymax);
        line(img,h,w,x1,y1,x2,y2,0,0);
        if (useSmall==-1) line(img,h,w,x1,y1,x1,y1,2,0);
        if (isDigits(inited)==1){
            y2B = 0.1 + 0.8*((accs2[i]-min)/ymax);
            y1B = 0.1 + 0.8*((accs2[i-1]-min)/ymax);
            line(img,h,w,x1,y1B,x2,y2B,0,4);
            if (useSmall==-1) line(img,h,w,x1,y1B,x1,y1B,2,4);
        }        
    }
    if (useSmall==-1) line(img,h,w,x2,y2,x2,y2,2,0);
    if (isDigits(inited)==1 && useSmall==-1) line(img,h,w,x2,y2B,x2,y2B,2,4);
    int d=3, ww=100, hh=10, ss=32, dy=7; if (useSmall==1) {d=1; ww=10; hh=1; ss=6; dy=0;}
    float scale = (max-min)/0.1;
    for (i=10;i>=0;i--){
        if (i==0 && max==1.0) dy=0;
        printInt((i+1)*ss,dy,img,h,w,(int)(100*max-scale*i),0,d);
    }
    printStr(hh,ww,img,h,w,"accuracy",0,d);
    websetcolors(8,red,green,blue,5);
    webimagedisplay(w,h,img,5);
    showDig[2][0]=0;
}

/**********************************************************************/
/*      DISPLAY PROGRESS                                              */
/**********************************************************************/
void displayConfusion(int (*confusion)[10]){
    // DISPLAY CONFUSION MATRIX FOR EITHER NN TRAIN OR KNN VALIDATION
    int i,j;
    int *img = (int*)image2;
    int w=120, h=80, t=1;
    img = (int*) image;
    w = 600;
    h = 400;
    t = 3;
    for (i=0;i<w*h;i++) img[i]=6;
    for (i=1;i<12;i++){
        line(img,h,w,i*0.135,0,i*0.135,0.9091,1,7);
        line(img,h,w,0.1364,(i-1)*0.09,1.5,(i-1)*0.09,1,7);
    }
    for (i=0;i<10;i++)
    for (j=0;j<10;j++)
        printInt((i+1)*36+12,(j+1)*54+4,img,h,w,confusion[i][j],4,3);
    for (i=0;i<10;i++){
        printInt((i+1)*36+12,20,img,h,w,i,0,3);
        printInt(12,(i+1)*54+20,img,h,w,i,0,3);
    }
    websetcolors(8,red,green,blue,3);
    webimagedisplay(w,h,img,3);
    showCon = 1;
    showDig[0][0]=0;
}

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
float ReLU(float x){
    if (x>0) return x;
    else return 0;
}

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
float TanH(float x){
	return 2.0/(1.0+exp(-2*x))-1.0;
}

/**********************************************************************/
/*      KNN                                                           */
/**********************************************************************/
int singleKNN(int x, int k, int d, int p, int train, int big, int disp){
    // PERFORM KNN ON A SINGLE IMAGE
    char buffer[80];
    int i,j,j2,c,max=0,imax=0,s=0,sz=196;
    float dist[k], dd;
    int idx[k];
    if (big==1) sz = 784;
    float (*testImages3)[sz];
    float (*trainImages3)[sz];
    if (big==0){
        if (train==1) {
            testImages3 = trainImages2;
            trainImages3 = trainImages2;
        }
        else {
            testImages3 = testImages2;
            trainImages3 = trainImages2;
        }
    }
    else if (big==1){
        if (train==1) {
            testImages3 = trainImages;
            trainImages3 = trainImages;
        }
        else {
            testImages3 = testImages;
            trainImages3 = trainImages;
        }
    }
    int votes[10];
        for (j=0;j<k;j++){
            idx[j] = j;
            dist[j] = distance( testImages3[x], trainImages3[j],sz,d);
            if (train==1 && x==j) dist[j] = 1e9;
        }   
        fakeHeap(dist,idx,k);
        for (j=k;j<trainSizeI;j++){
            dd = distance( testImages3[x], trainImages3[j],sz,d);
            if (train==1 && x==j) dd = 1e9;
            if (dd<dist[0]){
                dist[0] = dd;
                idx[0] = j;
                fakeHeap(dist,idx,k);
            }
        }
        max = 0;
        for (j=0;j<10;j++) votes[j]=0;
        for (j=0;j<k;j++){
            c = trainDigits[idx[j]];
            votes[c]++;
            if (votes[c]>max){
                max = votes[c];
                imax = c;
            }
        }
        if (disp==1){
            sortHeap(dist,idx,k);
            int dgs[k+1];
            dgs[0] = x;
            if (train==0){
                for (i=0;i<196;i++) trainImages2[trainSizeE][i] = testImages2[x][i];
                for (i=0;i<784;i++) trainImages[trainSizeE][i] = testImages[x][i];
                trainDigits[trainSizeE] = 0;
                dgs[0] = trainSizeE;
            }
            for (i=0;i<k;i++) dgs[i+1] = idx[i];
            displayDigits(dgs,k+1,p,1,0,0,big);
        }
        return imax;
}

/**********************************************************************/
/*      KNN                                                           */
/**********************************************************************/
void *runKNN(void *arg){
    // PERFORM KNN ON ENTIRE VALIDATION SET
    // I SHOULD UTILIZE SINGLE KNN FUNCTION
    int y2=pass[1], y=pass[1], k=pass[2], d=pass[3], z=pass[4], big=pass[0];
    if (validSetSize==0){
        webwriteline("Load images first. Click load.");
        working=0;
        return NULL;
    }
    showCon = 1;
    char buffer[80];
    sprintf(buffer,"Begin kNN with k=%d, L%d norm, big=%d",k,d,big);
    webwriteline(buffer);
    int dp = 0;
    int i,j,j2,c,max=0,imax=0,s=0;
    int trainSize = trainSetSize;
    int testSize = validSetSize;
    float dist[k], dd;
    int idx[k], dgs[24];
    int votes[10];
    int confusion[10][10]={{0}};
    for (i=0;i<10;i++) for (j=0;j<10;j++) for (j2=0;j2<maxCD;j2++) cDigits[i][j][j2]= -1;
    
    int size = 196; if (big==1 || big==2) size = 784;
    float (*trainImages3)[size];
    if (big==1 || big==2) trainImages3 = trainImages;
    else trainImages3 = trainImages2;
    if (big==2) size = trainColumns;
    
    for (i=0;i<validSetSize;i++){
        for (j=0;j<k;j++){
            idx[j] = j;
            dist[j] = distance( trainImages3[validSet[i]], trainImages3[trainSet[j]],size,d);
        }   
        fakeHeap(dist,idx,k);
        for (j=k;j<trainSetSize;j++){
            dd = distance( trainImages3[validSet[i]], trainImages3[trainSet[j]],size,d);
            if (dd<dist[0]){
                dist[0] = dd;
                idx[0] = j;
                fakeHeap(dist,idx,k);
            }
        }
        max = 0;
        for (j=0;j<10;j++) votes[j]=0;
        for (j=0;j<k;j++){
            c = trainDigits[trainSet[idx[j]]];
            votes[c]++;
            if (votes[c]>max){
                max = votes[c];
                imax = c;
            }
        }
        c = trainDigits[validSet[i]];
        if (c==imax) s++;
        cDigits[c][imax][ confusion[c][imax]%maxCD ] = validSet[i];
        confusion[c][imax]++;

        if (i%y2==0) {
            dp=0;
            if (showCon==1) displayConfusion(confusion);
        }
        if (c==imax && dp<4){
            int kk = 5; if (k<kk) kk=k;
            dgs[dp*6] = validSet[i];
            for (j=0;j<kk;j++) dgs[dp*6+j+1] = trainSet[idx[j]];
            for (j=kk;j<5;j++) dgs[dp*6+j+1] = -1;
            if (dp==3) displayDigits(dgs,24,5,1,0,0,2);
            dp++;
        }
        if (i==0 || (i+1)%y==0){
            sprintf(buffer,"i=%d, accuracy = %.2f",i+1,100.0*s/(i+1));
            if (z==1) webwriteline(buffer);
            else printf("%s\n",buffer);
        }
        if (working==0){
            webwriteline("learning stopped early");
            pthread_exit(NULL);
        }
    }
    sprintf(buffer,"i=%d, k=%d, Accuracy = %.2f",validSetSize,k,100.0*s/validSetSize);
    if (z==1) webwriteline(buffer);
    else printf("%s\n",buffer);
    webwriteline("Done");
    working=0;
    return NULL;
}

/**********************************************************************/
/*      KNN                                                           */
/**********************************************************************/
void fakeHeap(float *dist,int *idx,int k){
    // EMULATES A HEAP BY PUTTING MAX ITEM FIRST
    float max = 0.0;
    int imax = 0;
    for (int i=0;i<k;i++)
    if (dist[i]>max){
        max = dist[i];
        imax = i;
    }
    float tempDist;
    int tempIdx;
    tempDist = dist[0];
    tempIdx = idx[0];
    dist[0] = dist[imax];
    idx[0] = idx[imax];
    dist[imax] = tempDist;
    idx[imax] = tempIdx;
}

/**********************************************************************/
/*      KNN                                                           */
/**********************************************************************/
void sortHeap(float *dist, int *idx, int k){
    // SORTS HEAP
    float dist2[k];
    int idx2[k];
    int i,j;
    float min;
    int imin = 0;
    for (j=0;j<k;j++){
        min = 1e6;
        for (i=0;i<k;i++){
            if (dist[i]<min){
                imin = i;
                min = dist[i];
            }
        }
        dist2[j] = dist[imin];
        dist[imin]=1e6;
        idx2[j] = idx[imin];
     }
     for (i=0;i<k;i++){
         dist[i] = dist2[i];
         idx[i] = idx2[i];
     }
}

/**********************************************************************/
/*      KNN                                                           */
/**********************************************************************/
float distance(float *digitA, float *digitB, int n, int x){
    // RETURNS DISTANCE BETWEEN DIGITS USING L-X NORM
    int i=0;
    float dist = 0.0;
    for (i=0;i<n;i++){
        if (x==1) dist += fabs(digitA[i] - digitB[i]);
        else if (x==2) dist += pow(digitA[i] - digitB[i],2);
    }
    return dist;
}

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
void *runBackProp(void *arg){
    // TRAINS NEURAL NETWORK WITH TRAINING DATA
    time_t start,stop;
    showEnt = 1; showAcc = 1;
    char buffer[80];
    int i, x = pass[0], y = pass[1], z = pass[2];
    int p, confusion[10][10]={{0}};
    if (layers[0]==NULL){
        initNet(1);
        if (z==1) {
            sprintf(buffer,"Assuming NN=1 with Xavier init scaled=%.3f",weightScale);
            webwriteline(buffer);
        }
        int len = sprintf(buffer,"Architecture (%d",layerSizes[0]);
        for (i=1;i<10;i++) len += sprintf(buffer+len,"-%d",layerSizes[i]);
        sprintf(buffer+len,")");
        if (z==1) webwriteline(buffer);
    }
    // LEARN DIGITS
    int trainSize = trainSetSize;
    int testSize = validSetSize;
    if (isDigits(inited)==1) {
        websetmode(2);
        showCon=1;
    }
    else { // LEARN DOTS
        trainSize = trainSizeD;
        testSize = 0;
        websetmode(dotsMode);
    }
    if (trainSize==0){
        if (isDigits(inited)==1) webwriteline("Load images first. Click load.");
        else webwriteline("Create training dots first. Click dots inside pane to the right.");
        working=0; websetmode(2);
        return NULL;
    }
    // ALLOCATE MEMORY FOR ENTORPY AND ACCURACY HISTORY
    if (ents!=NULL){
        free(ents); free(ents2); free(accs); free(accs2);
    }
    ents = (float*)malloc( (int)(x/y+1) * sizeof(float) );
    ents2 = (float*)malloc( (int)(x/y+1) * sizeof(float) );
    accs = (float*)malloc( (int)(x/y+1) * sizeof(float) );
    accs2 = (float*)malloc( (int)(x/y+1) * sizeof(float) );
    int entSize = 0, accSize = 0, ent2Size = 0, acc2Size = 0;
    int j,j2,k,s,s2,b;
    float entropy,entropy2,ent;
    time(&start);
    // PERFORM X TRAINING EPOCHS
    for (j=0;j<x;j++){
        s = 0; entropy = 0.0;
        if (isDigits(inited)!=1) trainSize = trainSizeD;
        for (i=0;i<trainSize;i++){
            //if (i%100==0) printf("x=%d, i=%d\n",j,i);
            if (isDigits(inited)==1) b = backProp(trainSet[i],&ent,j); // LEARN DIGITS
            else b = backProp(i,&ent,0); // LEARN DOTS
            if (b==-1) {
                if (z==1) webwriteline("Exploded. Lower learning rate.");
                else printf("Exploded. Lower learning rate.\n");
                working=0; websetmode(2);
                return NULL;
            }
            s += b;
            entropy += ent;
            if (working==0){
                webwriteline("learning stopped early");
                pthread_exit(NULL);
            }
        }
        entropy = entropy / trainSize;
        s2 = 0; entropy2 = 0.0;
        for (i=0;i<10;i++) for (k=0;k<10;k++) confusion[i][k]=0;
        for (i=0;i<10;i++) for (j2=0;j2<10;j2++) for (k=0;k<maxCD;k++) cDigits[i][j2][k]= -1;
        for (i=0;i<testSize;i++){
            p = forwardProp(validSet[i],0,1);
            if (p==-1) {
                if (z==1) webwriteline("Test exploded.");
                else printf("Test exploded.\n");
                working=0; websetmode(2);
                return NULL;
            }
            if (p==trainDigits[validSet[i]]) s2++;
            cDigits[trainDigits[validSet[i]]][p][ confusion[trainDigits[validSet[i]]][p]%maxCD ] = validSet[i];
            confusion[trainDigits[validSet[i]]][p]++;
            if (layers[9][p]==0){
                if (z==1) webwriteline("Test vanished.");
                else printf("Test vanished.\n");
                working=0; websetmode(2);
                return NULL;
            }
            entropy2 -= log(layers[9][p]);
            if (working==0){
                webwriteline("learning stopped early");
                pthread_exit(NULL);
            }
        }
        entropy2 = entropy2 / testSize;
        if (j==0 || (j+1)%y==0){
            ents[entSize++] = entropy;
            accs[accSize++] = (float)s/trainSize;
            if (isDigits(inited)==1) {
                accs2[acc2Size++] = (float)s2/testSize;
                ents2[ent2Size++] = entropy2;
            }
            time(&stop);
            sprintf(buffer,"i=%d acc=%d/%d, ent=%.4f, lr=%.1e",j+1,s,trainSize,entropy,an*pow(decay,j));
            if (isDigits(inited)==1 && testSize>0) sprintf(buffer,"i=%d train=%.2f ent=%.4f,valid=%.2f ent=%.4f (%.0fsec) lr=%.1e",
                j+1,100.0*s/trainSize,entropy,100.0*s2/testSize,entropy2,difftime(stop,start),an*pow(decay,j));
            else if (isDigits(inited)==1 && testSize==0) sprintf(buffer,"i=%d train=%.2f ent=%.4f (%.0fsec) lr=%.1e",
                j+1,100.0*s/trainSize,entropy,difftime(stop,start),an*pow(decay,j));
            time(&start);
            if (z==1) webwriteline(buffer);
            else printf("%s\n",buffer);
            if (z==1 && isDigits(inited)!=1) displayClassify(0);
            if (z==1 && showEnt==1) displayEntropy(ents,entSize,ents2,y);
            if (z==1 && showAcc==1) displayAccuracy(accs,accSize,accs2,y);
            if (z==1 && isDigits(inited)==1 && showCon==1)  displayConfusion(confusion);
        }
        if (working==0){
            webwriteline("learning stopped early");
            pthread_exit(NULL);
        }
        if (isDigits(inited)==1 && randomizeDescent==1) randomizeTrainSet();
    }
    webwriteline("Done");
    working=0; websetmode(2);
    return NULL;
}

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
void randomizeTrainSet(){
    // RANDOMIZES INDICES IN TRAINING SET
    int i, temp, x;
    for (i=0;i<trainSetSize;i++){
        x = (int)(trainSetSize * ((float)rand()/(float)RAND_MAX) - 1);
        temp = trainSet[i];
        trainSet[i] = trainSet[x];
        trainSet[x] = temp;
    }
}

/**********************************************************************/
/*      PREDICT                                                       */
/**********************************************************************/
void writePredictFile(int NN, int k, int d, int y, int big){
    // PREDICTS TEST IMAGES AND WRITES RESULTS TO FILE
    char buffer[80];
    if (NN==1){
        for (int i=0;i<testSizeI;i++) {
            testDigits[i] = forwardProp(i,0,0);
            if (i==0 || (i+1)%y==0){
                sprintf(buffer,"i=%d",i+1);
                webwriteline(buffer);
            }
        }
        writeFile();
    }
    else if (working==1){
        webwriteline("wait until learning ends");
        return;
    }
    else{
        sprintf(buffer,"predict kNN with k=%d, L%d norm, big=%d",k,d,big);
        webwriteline(buffer);
        pass[0]=k; pass[1]=d; pass[2]=y; pass[3]=big; working=1;
        pthread_create(&workerThread,&stackSizeAttribute,predictKNN,NULL);
    }
}

/**********************************************************************/
/*      PREDICT                                                       */
/**********************************************************************/
void writeFile(){
    // FUNCTION TO WRITE PREDICTIONS TO FILE
    int in = layerSizes[10-numLayers], offset=1;
    char buffer[80];
    FILE *fp;
    fp = fopen(spGet("fileName"),"w");
    char colA[40], colB[40];
    strcpy(colA,spGet("col1Name"));
    strcpy(colB,spGet("col2Name"));
    sprintf(buffer,"%s,%s\r\n",colA,colB);
    offset = ipGet("row1Num");
    fputs(buffer,fp);
    for (int i=0;i<testSizeI;i++){
        sprintf(buffer,"%d,%d\r\n",i+offset,testDigits[i]);
        fputs(buffer,fp);
    }
    fclose(fp);
    sprintf(buffer,"Wrote predictions to %s\n",spGet("fileName"));
    webwriteline(buffer);
}

/**********************************************************************/
/*      SPECIAL AND/OR DEBUGGING                                      */
/**********************************************************************/
void writeFile2(){
    // FUNCTION TO WRITE DEBUG WEIGHTS
    char buffer[80];
    FILE *fp;
    fp = fopen(weightsFile1,"w");
    sprintf(buffer,"var weightsHidden = [%.6f",weights[8][0]);
    fputs(buffer,fp);
    for (int i=1;i<layerSizes[8]*(layerSizes[7]+1);i++){
        sprintf(buffer,", %.6f",weights[8][i]);
        fputs(buffer,fp);
    }
    fputs("];",fp);
    fclose(fp);
    sprintf(buffer,"Wrote hidden weights for %d-%d to %s\n",layerSizes[7],layerSizes[8],weightsFile1);
    webwriteline(buffer);
    fp = fopen(weightsFile2,"w");
    sprintf(buffer,"var weightsOutput = [%.6f",weights[9][0]);
    fputs(buffer,fp);
    for (int i=1;i<layerSizes[9]*(layerSizes[8]+1);i++){
        sprintf(buffer,", %.6f",weights[9][i]);
        fputs(buffer,fp);
    }
    fputs("];",fp);
    fclose(fp);
    sprintf(buffer,"Wrote output weights for %d-%d to %s\n",layerSizes[8],layerSizes[9],weightsFile2);
    webwriteline(buffer);
}

/**********************************************************************/
/*      KNN                                                           */
/**********************************************************************/
void *predictKNN(void *arg){
    // FUNCTION TO MAKE PREDICTIONS TO WRITE TO FILE
    int k=pass[0], d=pass[1], y=pass[2], big=pass[3];
    char buffer[80];
    float dist[k], dd;
    int c, i, max, imax, j, idx[k], votes[10];
    int size = 196; if (big==1 || big==2) size = 784;
    float (*imageTrain)[size];
    if (big==1 || big==2) imageTrain = trainImages;
    else imageTrain = trainImages2;
    float (*imageTest)[size];
    if (big==1 || big==2) imageTest = testImages;
    else imageTest = testImages2;
    if (big==2) size = trainColumns;
    for (i=0;i<testSizeI;i++){
        for (j=0;j<k;j++){
            idx[j] = j;
            dist[j] = distance( imageTest[i], imageTrain[j],size,d);
        }   
        fakeHeap(dist,idx,k);
        for (j=k;j<trainSizeI;j++){
            dd = distance( imageTest[i], imageTrain[j],size,d);
            if (dd<dist[0]){
                dist[0] = dd;
                idx[0] = j;
                fakeHeap(dist,idx,k);
            }
        }
        max = 0;
        for (j=0;j<10;j++) votes[j]=0;
        for (j=0;j<k;j++){
            c = trainDigits[idx[j]];
            votes[c]++;
            if (votes[c]>max){
                max = votes[c];
                imax = c;
            }
        }
        testDigits[i] = imax;
        if (i==0 || (i+1)%y==0){
            sprintf(buffer,"i=%d",i+1);
            webwriteline(buffer);
        }
        if (working==0){
            webwriteline("predict kNN stopped early");
            pthread_exit(NULL);
        }
    }
    writeFile();
    working=0;
    return NULL;
}

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
int backProp(int x, float *ent, int ep){
    // BACK PROPAGATION WITH 1 TRAINING IMAGE
    int i = 0, j, k, r = 0, d=0, rot=0, hres=0, lres=1;
    float der=1.0, xs=0.0, ys=0.0, extra=0.0, sc=1.0;
    int dc, a, a2, i2, j2, i3, j3, pmax, imax, jmax;
    int temp, temp2;
    // DATA AUGMENTATION
    if (augmentRatio>0.0)
    if ( (float)rand()/(float)RAND_MAX <= augmentRatio ){
        if (augmentAngle>0.0)
            rot = (int)(2.0 * augmentAngle * (float)rand()/(float)RAND_MAX - augmentAngle);
        if (augmentDx>0.0)
            xs = (2.0 * augmentDx * (float)rand()/(float)RAND_MAX - augmentDx);
        if (augmentDy>0.0)
            ys = (2.0 * augmentDy * (float)rand()/(float)RAND_MAX - augmentDy);
        if (augmentScale>0.0)
            sc = 1.0 + 2.0 * augmentScale * (float)rand()/(float)RAND_MAX - augmentScale;
        if (layerSizes[10-numLayers]==784){hres=1;lres=0;}
        dataAugment(x,rot,sc,xs,ys,-1,hres,lres,1);
        x = trainSizeE;
    }
    // FORWARD PROP FIRST
    int p = forwardProp(x,1,1);
    if (p==-1) return -1; // GRADIENT EXPLODED
    // CORRECT PREDICTION?
    int y;
    if (isDigits(inited)==1) y = trainDigits[x];
    else y = trainColors[x];
    if (p==y) r=1;
    // OUTPUT LAYER - CALCULATE ERRORS
    for (i=0;i<layerSizes[9];i++){
        errors[9][i] = an * (0 - layers[9][i]) * pow(decay,ep);
        if (i==y) {
            errors[9][i] = an * (1  - layers[9][i]) * pow(decay,ep);
            if (layers[9][i]==0) return -1; // GRADIENT VANISHED
            *ent = -log(layers[9][i]);
        }
    }
    // HIDDEN LAYERS - CALCULATE ERRORS
    for (k=8;k>10-numLayers;k--){
    if (layerType[k+1]==0) // FEEDS INTO FULLY CONNECTED
    for (i=0;i<layerSizes[k]*layerChan[k];i++){
        errors[k][i] = 0.0;
        if (dropOutRatio==0.0 || DOdense==0 || dropOut[k][i]==1){ // dropout
            if (activation==2) der = (layers[k][i]+1)*(1-layers[k][i]); //TanH derivative
            if (activation==2 || layers[k][i]>0){ //this is ReLU derivative
                temp = layerSizes[k]*layerChan[k]+1;
                for (j=0;j<layerSizes[k+1];j++)
                    errors[k][i] += errors[k+1][j]*weights[k+1][j*temp+i]*der;
            }
        }
    }
    else if (layerType[k+1]==1){ // FEEDS INTO CONVOLUTION
        for (i=0;i<layerSizes[k]*layerChan[k];i++) errors[k][i] = 0.0;
        dc = 0; if (layerPad[k+1]==1) dc = layerConv[k+1]/2;
        for (a=0;a<layerChan[k+1];a++)
        for (i=0;i<layerWidth[k+1];i++)
        for (j=0;j<layerWidth[k+1];j++){
            temp = a*(layerConvStep[k+1]+1);
            temp2 = a*layerSizes[k+1] + i*layerWidth[k+1] + j;
            for (a2=0;a2<layerChan[k];a2++)
            for (i2=0;i2<layerConv[k+1];i2++)
            for (j2=0;j2<layerConv[k+1];j2++){
                i3 = i + i2 - dc;
                j3 = j + j2 - dc;
                if (activation==2) der = (layers[k][a2*layerSizes[k] + i3*layerWidth[k] + j3]+1)*(1-layers[k][a2*layerSizes[k] + i3*layerWidth[k] + j3]); //TanH
                if (activation==2 || layers[k][a2*layerSizes[k] + i3*layerWidth[k] + j3]>0) // this is ReLU derivative
                if (i3>=0 && i3<layerWidth[k] && j3>=0 && j3<layerWidth[k]) // padding
                errors[k][a2*layerSizes[k] + i3*layerWidth[k] + j3] +=
                    weights[k+1][temp + a2*layerConvStep2[k+1] + i2*layerConv[k+1] +j2]
                    * errors[k+1][temp2] * der;
            }
        }
        if (dropOutRatio>0.0 && DOconv==1) // dropout
        for (i=0;i<layerSizes[k]*layerChan[k];i++) errors[k][i] = errors[k][i] * dropOut[k][i];
    }
    else if (layerType[k+1]==2){ // FEEDS INTO MAX POOLING
        for (i=0;i<layerSizes[k]*layerChan[k];i++) errors[k][i] = 0.0;
        for (a=0;a<layerChan[k];a++)
        for (i=0;i<layerWidth[k+1];i++)
        for (j=0;j<layerWidth[k+1];j++){
            pmax = -1e6;
            for (i2=0;i2<layerConv[k+1];i2++)
            for (j2=0;j2<layerConv[k+1];j2++)
                if (layers[k][a*layerSizes[k] + (i*layerConv[k+1]+i2)*layerWidth[k] + j*layerConv[k+1]+j2]>pmax){
                    pmax = layers[k][a*layerSizes[k] + (i*layerConv[k+1]+i2)*layerWidth[k] + j*layerConv[k+1]+j2];
                    imax = i2;
                    jmax = j2;
                }
            errors[k][a*layerSizes[k] + (i*layerConv[k+1]+imax)*layerWidth[k] + j*layerConv[k+1]+jmax] =
                errors[k+1][a*layerSizes[k+1] + i*layerWidth[k+1] + j];
        }
        if (dropOutRatio>0.0 && DOpool==1) //dropout
        for (i=0;i<layerSizes[k]*layerChan[k];i++) errors[k][i] = errors[k][i] * dropOut[k][i];
    }
    }
    
    // UPDATE WEIGHTS - GRADIENT DESCENT
    int count = 0;
    for (k=11-numLayers;k<10;k++){
    if (layerType[k]==0){ // FULLY CONNECTED LAYER
        for (i=0;i<layerSizes[k];i++){
            temp = i*(layerSizes[k-1]*layerChan[k-1]+1);
            for (j=0;j<layerSizes[k-1]*layerChan[k-1]+1;j++)
                weights[k][temp+j] += errors[k][i]*layers[k-1][j];
        }
    }
    else if (layerType[k]==1){ // CONVOLUTION LAYER
        dc = 0; if (layerPad[k]==1) dc = layerConv[k]/2;
        for (a=0;a<layerChan[k];a++)
        for (i=0;i<layerWidth[k];i++)
        for (j=0;j<layerWidth[k];j++){
            temp = a*(layerConvStep[k]+1);
            temp2 = a*layerSizes[k] + i*layerWidth[k] + j;
            for (a2=0;a2<layerChan[k-1];a2++)
            for (i2=0;i2<layerConv[k];i2++)
            for (j2=0;j2<layerConv[k];j2++){
                i3 = i + i2 - dc;
                j3 = j + j2 - dc;
                if (i3>=0 && i3<layerWidth[k-1] && j3>=0 && j3<layerWidth[k-1])
                weights[k][temp + a2*layerConvStep2[k] + i2*layerConv[k] + j2] +=
                    errors[k][temp2] * layers[k-1][a2*layerSizes[k-1] + i3*layerWidth[k-1] + j3];
            }
            weights[k][(a+1)*(layerConvStep[k]+1)-1] += errors[k][a*layerSizes[k] + i*layerWidth[k] + j];
        }
    }
    
    }
    
    return r;
}

/**********************************************************************/
/*      NEURAL NETWORK                                                */
/**********************************************************************/
int forwardProp(int x, int dp, int train){
    // FORWARD PROPAGATION WITH 1 IMAGE
    int i,j,k,imax,dc;
    int a, a2, i2, j2, i3, j3;
    float sum, esum, max, rnd, pmax;
    int temp, temp2;
    // INPUT LAYER
    if (isDigits(inited)==1 && layerSizes[10-numLayers]==196){
        if (train==1) for (i=0;i<196;i++) layers[10-numLayers][i] = trainImages2[x][i];
        else for (i=0;i<196;i++) layers[10-numLayers][i] = testImages2[x][i];
    }
    else if (isDigits(inited)==1 && layerSizes[10-numLayers]==784){
        if (train==1) for (i=0;i<784;i++) layers[10-numLayers][i] = trainImages[x][i];
        else for (i=0;i<784;i++) layers[10-numLayers][i] = testImages[x][i];
    }
    else if (isDigits(inited)==1 && layerSizes[10-numLayers]==trainColumns){
        if (train==1) for (i=0;i<trainColumns;i++) layers[10-numLayers][i] = trainImages[x][i];
        else for (i=0;i<trainColumns;i++) layers[10-numLayers][i] = testImages[x][i];
    }
    else if (layerSizes[10-numLayers]==2)
        for (i=0;i<2;i++) layers[10-numLayers][i] = trainDots[x][i];
    
    // HIDDEN LAYERS
    for (k=11-numLayers;k<9;k++){
    
    // CALCULATE DROPOUT
    //if (dropOutRatio>0.0) // ALWAYS SET TO 1 TO BE SAFE
    for (i=0;i<layerSizes[k]*layerChan[k];i++) {
        dropOut[k][i] = 1;
        if (dropOutRatio>0.0 && dp==1) {
            rnd = (float)rand()/(float)RAND_MAX;
            if (rnd<dropOutRatio) dropOut[k][i] = 0;
        }
    }
    
    if (layerType[k]==0) // FULLY CONNECTED LAYER
    for (i=0;i<layerSizes[k];i++){
        if (dropOutRatio==0.0 || dp==0 || DOdense==0 || dropOut[k][i]==1){
            temp = i*(layerSizes[k-1]*layerChan[k-1]+1);
            sum = 0.0;
            for (j=0;j<layerSizes[k-1]*layerChan[k-1]+1;j++)
                sum += layers[k-1][j]*weights[k][temp+j];
            if (activation==1) layers[k][i] = ReLU(sum);
            else layers[k][i] = TanH(sum);
            //if (dropOutRatio>0.0 && dp==1) layers[k][i] = layers[k][i]  / (1-dropOutRatio);
            if (dropOutRatio>0.0 && dp==0 && DOdense==1) layers[k][i] = layers[k][i]  * (1-dropOutRatio);
        }
        else layers[k][i] = 0.0;
    }
    else if (layerType[k]==1){ // CONVOLUTION LAYER
        dc = 0; if (layerPad[k]==1) dc = layerConv[k]/2;
        for (a=0;a<layerChan[k];a++)
        for (i=0;i<layerWidth[k];i++)
        for (j=0;j<layerWidth[k];j++){
            temp = a*(layerConvStep[k]+1);
            sum = 0.0;
            for (a2=0;a2<layerChan[k-1];a2++)
            for (i2=0;i2<layerConv[k];i2++)
            for (j2=0;j2<layerConv[k];j2++){
                i3 = i + i2 - dc;
                j3 = j + j2 - dc;
                if (i3>=0 && i3<layerWidth[k-1] && j3>=0 && j3<layerWidth[k-1])
                sum += layers[k-1][a2*layerSizes[k-1] + i3*layerWidth[k-1] + j3] * weights[k][temp + a2*layerConvStep2[k] + i2*layerConv[k] + j2];
                else sum -= imgBias * weights[k][temp + a2*layerConvStep2[k] + i2*layerConv[k] + j2];
            }
            sum += weights[k][(a+1)*(layerConvStep[k]+1)-1];
            if (activation==1) layers[k][a*layerSizes[k] + i*layerWidth[k] + j] = ReLU(sum);
            else layers[k][a*layerSizes[k] + i*layerWidth[k] + j] = TanH(sum);
        }
        // APPLY DROPOUT
        if (dropOutRatio>0.0 && DOconv==1)
        for (i=0;i<layerSizes[k]*layerChan[k];i++){
            if (dp==0) layers[k][i] = layers[k][i]  * (1-dropOutRatio);
            else if (dp==1) layers[k][i] = layers[k][i]  * dropOut[k][i];
        }
    }
    else if (layerType[k]==2) // MAX POOLING LAYER
        for (a=0;a<layerChan[k];a++)
        for (i=0;i<layerWidth[k];i++)
        for (j=0;j<layerWidth[k];j++){
            pmax = -1e6;
            for (i2=0;i2<layerConv[k];i2++)
            for (j2=0;j2<layerConv[k];j2++)
                if (layers[k-1][a*layerSizes[k-1] + (i*layerConv[k]+i2)*layerWidth[k-1] + j*layerConv[k]+j2]>pmax)
                    pmax = layers[k-1][a*layerSizes[k-1] + (i*layerConv[k]+i2)*layerWidth[k-1] + j*layerConv[k]+j2];
            layers[k][a*layerSizes[k] + i*layerWidth[k] + j] = pmax;
        }
        // APPLY DROPOUT
        if (dropOutRatio>0.0 && DOpool==1)
        for (i=0;i<layerSizes[k]*layerChan[k];i++){
            if (dp==0) layers[k][i] = layers[k][i]  * (1-dropOutRatio);
            else if (dp==1) layers[k][i] = layers[k][i]  * dropOut[k][i];
        }
    }
    
    // OUTPUT LAYER - SOFTMAX ACTIVATION
    esum = 0.0;
    for (i=0;i<layerSizes[9];i++){
        sum = 0.0;
        for (j=0;j<layerSizes[8]+1;j++)
            sum += layers[8][j]*weights[9][i*(layerSizes[8]+1)+j];
        layers[9][i] = exp(sum);
        if (layers[9][i]>1e30) return -1; //GRADIENTS EXPLODED
        esum += layers[9][i];
    }
    
    // SOFTMAX FUNCTION
    max = layers[9][0]; imax=0;
    for (i=0;i<layerSizes[9];i++){
        if (layers[9][i]>max){
            max = layers[9][i];
            imax = i;
        }
        layers[9][i] = layers[9][i] / esum;
    }
    return imax;
}

/**********************************************************************/
/*      INIT NET                                                      */
/**********************************************************************/
void initArch(char *str, int x){
    // PARSES USER INPUT TO CREATE DESIRED NETWORK ARCHITECTURE
    //TODO: remove all spaces, check for invalid characters
    int i;
    char *idx = str, *idx2;
    while (idx[0]==' ' && idx[0]!=0) idx++;
    for (i=0;i<strlen(idx);i++) str[i]=idx[i];
    if (str[0]==0) {str[0]='0'; str[1]=0;}
    if (str[0]>='0' && str[0]<='9'){
        layerSizes[x] = atoi(str);
        layerConv[x] = 0;
        layerChan[x] = 1;
        layerPad[x] = 0;
        layerWidth[x] = (int)sqrt(layerSizes[x]);
        if (layerWidth[x]*layerWidth[x]!=layerSizes[x]) layerWidth[x]=1;
        layerStride[x] = 1;
        layerConvStep[x] = 0;
        layerConvStep2[x] = 0;
        layerType[x] = 0;
    }
    else if (str[0]=='c' || str[0]=='C'){
        int more = 1;
        str[0]='C';
        idx = str+1;
        while(*idx!=':' && *idx!='-' && *idx!=0) idx++;
        if (*idx==0) more = 0; *idx = 0;
        layerConv[x] = atoi(str+1);
        layerChan[x] = 1;
        layerPad[x] = 0;
        //layerWidth[x] = layerWidth[x-1];
        layerWidth[x] = layerWidth[x-1]-layerConv[x]+1;
        if (more==1){
            *idx = ':';
            idx++; idx2 = idx;
            while(*idx!=':' && *idx!='-' && *idx!=0) idx++;
            if (*idx==0) more = 0; *idx = 0;
            layerChan[x] = atoi(idx2);
            if (more==1){
                *idx = ':';
                idx++; idx2 = idx;
                while(*idx!=':' && *idx!='-' && *idx!=0) idx++;
                if (*idx==0) more = 0; *idx = 0;
                layerPad[x] = atoi(idx2);
                if (layerPad[x]==1)
                    layerWidth[x] = layerWidth[x-1];
                    //layerWidth[x] = layerWidth[x-1]-layerConv[x]+1;
            }
        }
        layerSizes[x] = layerWidth[x] * layerWidth[x];
        layerConvStep2[x] = layerConv[x] * layerConv[x];
        layerConvStep[x] = layerConvStep2[x] * layerChan[x-1];
        layerStride[x] = 1;
        layerType[x] = 1;
    }
    else if (str[0]=='p' || str[0]=='P'){
        str[0]='P';
        layerConv[x] = atoi(str+1);
        int newWidth = layerWidth[x-1]/layerConv[x];
        layerSizes[x] = newWidth * newWidth;
        layerChan[x] = layerChan[x-1];
        layerPad[x] = 0;
        layerWidth[x] = newWidth;
        layerStride[x] = 1;
        layerConvStep[x] = 0;
        layerConvStep2[x] = 0;
        layerType[x] = 2;
    }
    strcpy(layerNames[x],str);
}

/**********************************************************************/
/*      INIT NET                                                      */
/**********************************************************************/
void initNet(int t){
    // ALLOCATION MEMORY AND INITIALIZE NETWORK WEIGHTS
     int i,j, same=1, LL, dd=9;
     char buf[10], buf2[20];
     if (t==0){
         for (i=0;i<10;i++) {
             strcpy(nets[0][i],"0");
             layerType[i] = 0;
         }
         for (i=9;i>=0;i--){
             sprintf(buf,"L%d",i);
             strcpy (buf2,spGet(buf));
             buf2[19]=0;
             if (buf2[0]!=0 && buf2[0]!='0'){
                 if (strcmp(buf2,nets[0][dd])!=0) same=0;
                 strcpy(nets[0][dd--],buf2);
             }
         }
         if (numLayers!=9-dd) same=0;
     }
     // FREE OLD NET
     if ( (t!=inited && layers[0]!=NULL) || (t==0 && same==0) ){
         free(layers[0]);
         free(errors[0]);
         for (i=1;i<10;i++){
             free(layers[i]);
             free(dropOut[i]);
             free(errors[i]);
             free(weights[i]);
         } 
         layers[0] = NULL;
     }
     // SET NEW NET ARCHITECTURE
     numLayers = 0;
     for (i=0;i<10;i++) {
         initArch(nets[t][i],i);
         sprintf(buf,"L%d",i);
         spSet(buf,nets[t][i]);
         if (numLayers==0 && layerSizes[i]!=0) numLayers = 10-i;
     }
     webupdate(ip,rp,sp);
     //printf("\n");
    
     // ALOCATE MEMORY
     if (layers[0]==NULL){
         layers[0] = (float*)malloc((layerSizes[0]+1) * sizeof(float));
         errors[0] = (float*)malloc(layerSizes[0] * sizeof(float));
         for (i=1;i<10;i++){
             layers[i] = (float*)malloc((layerSizes[i] * layerChan[i] + 1) * sizeof(float));
             dropOut[i] = (int*)malloc((layerSizes[i] * layerChan[i] + 1) * sizeof(int));
             //printf("setting dropOut i=%d to %d\n",i,(layerSizes[i] * layerChan[i] + 1));
             errors[i] = (float*)malloc((layerSizes[i] * layerChan[i] + 1) * sizeof(float));
             if (layerType[i]==0) // FULLY CONNECTED
                 weights[i] = (float*)malloc(layerSizes[i] * (layerSizes[i-1]*layerChan[i-1]+1) * sizeof(float));
             else if (layerType[i]==1) // CONVOLUTION
                 weights[i] = (float*)malloc((layerConvStep[i]+1) * layerChan[i] * sizeof(float));
             else if (layerType[i]==2) // POOLING
                 weights[i] = (float*)malloc( sizeof(float));
         }
     }
     // RANDOMIZE WEIGHTS AND BIAS
     float scale;
     for (i=0;i<10;i++) layers[i][layerSizes[i] * layerChan[i]]=1.0;
     for (j=1;j<10;j++){
         scale = 1.0;
         if (layerSizes[j-1]!=0){
              // XAVIER INITIALIZATION (= SQRT( 6/(N_in + N_out) ) ) What is N_out to MaxPool ??
              if (layerType[j]==0){ // FC LAYER
                if (layerType[j+1]==0)
                    scale = 2.0 * sqrt(6.0/ ( layerSizes[j-1]*layerChan[j-1] + layerSizes[j] ));
                else if (layerType[j+1]==1) // impossible
                    scale = 2.0 * sqrt(6.0/ ( layerSizes[j-1]*layerChan[j-1] + layerConvStep[j+1] ));
                else if (layerType[j+1]==2) // impossible
                    scale = 2.0 * sqrt(6.0/ ( layerSizes[j-1]*layerChan[j-1] + layerSizes[j-1]*layerChan[j-1] ));
              }
              else if (layerType[j]==1){ // CONV LAYER
                if (layerType[j+1]==0)
                    scale = 2.0 * sqrt(6.0/ ( layerConvStep[j] + layerSizes[j]*layerChan[j] ));
                else if (layerType[j+1]==1)
                    scale = 2.0 * sqrt(6.0/ ( layerConvStep[j] + layerConvStep[j+1] ));
                else if (layerType[j+1]==2)
                    scale = 2.0 * sqrt(6.0/ ( layerConvStep[j] + layerConvStep[j] ));
              }
              //if (activation==1 && j!=9) scale *= sqrt(2.0); // DO I WANT THIS? INPUT ISN'T MEAN=0
              //printf("Init layer %d: LS=%d LC=%d LCS=%d Scale=%f\n",j,layerSizes[j],layerChan[j],layerConvStep[j],scale);
              if (j!=9) scale *= weightScale;
         }
         if (layerType[j]==0){ // FULLY CONNECTED
            for (i=0;i<layerSizes[j] * (layerSizes[j-1]*layerChan[j-1]+1);i++)
                weights[j][i] = scale * ( (float)rand()/(float)RAND_MAX - 0.5 );
                //weights[j][i] = 0.1;
             for (i=0;i<layerSizes[j];i++) // set biases to zero
                weights[j][(layerSizes[j-1]*layerChan[j-1]+1)*(i+1)-1] = 0.0;
         }
         else if (layerType[j]==1){ // CONVOLUTION
            for (i=0;i<(layerConvStep[j]+1) * layerChan[j];i++)
                weights[j][i] = scale * ( (float)rand()/(float)RAND_MAX - 0.5 );
            for (i=0;i<layerChan[j];i++) // set conv biases to zero
                weights[j][(layerConvStep[j]+1)*(i+1)-1] = 0.0;
         }
     }

     inited = t;
     if (isDigits(inited)!=1) {
         showCon = 0;
         showDig[0][0] = 0;
         updateImage();
     }
}

/**********************************************************************/
/*      LOAD DATA                                                     */
/**********************************************************************/
int loadTrain(int ct, double testProp, int sh, float imgScale, float imgBias){
    // LOAD TRAINING DATA FROM FILE
    if (ct<=0) ct=1e6;
    int i, len = 0, lines=1, lines2=1;
    float rnd;
    // READ IN TRAIN.CSV
    char buffer[1000000];
    char name[80] = "train.csv";
    strcpy(name,spGet("dataFile"));
    if (access(name,F_OK)!=0) sprintf(name,"../%s",spGet("dataFile"));
    if (access(name,F_OK)==0){
        data = (char*)malloc((int)fsize(name)+1);
        FILE *fp;
        fp = fopen(name,"r");
        while (fgets(buffer, 1000000, fp)) {
            len += sprintf(data+len,"%s",buffer);
            //lines++;
        }
        fclose(fp);
    }
    else {
        sprintf(buffer,"ERROR: File %s not found.",name);
        webwriteline(buffer);
        return 0;
    }
    // COUNT LINES
    for (i=0;i<len;i++){
        if (data[i]=='\n') lines++;
        if (data[i]=='\r') lines2++;
    }
    if (lines2>lines) lines=lines2;
    // ALLOCATE MEMORY
    if (trainImages!=NULL){
        free(trainImages);
        free(trainImages2);
        free(trainDigits);
        free(trainSet);
        free(validSet);
    }
    trainImages = malloc(784 * (lines+extraTrainSizeI) * sizeof(float));
    trainImages2 = malloc(196 * (lines+extraTrainSizeI) * sizeof(float));
    trainDigits = malloc(lines * sizeof(int));
    trainSet = malloc(lines * sizeof(int));
    validSet = malloc(lines * sizeof(int));
    // DECODE COMMA SEPARATED ROWS
    int j = 0, k = 0, c = 0, mark = -1;
    int d = 0, j1,j2;
    while (data[j]!='\n' && data[j]!='\r'){
        if (data[j]==',') c++;
        j++;
    }
    if (data[j]!='\n' || data[j]!='\r') j++;
    trainColumns = c;
    c = 0; i = 0;
    if (sh==1) i = j+1;
    while(i<len && k<ct){
    	  j = i; while (data[j]!=',' && data[j]!='\r' && data[j]!='\n') j++;
    	  if (data[j]=='\n' || data[j]=='\r') mark = 1;
          data[j] = 0;
    	  d = atof(data+i);
          if (mark == -1){
              trainDigits[k] = (int)d;
              mark = 0;
          }
          else if (mark==0) {
              trainImages[k][c] = d/imgScale - imgBias;
              c++;
          }
          if (mark>=1){
              trainImages[k][c] = d/imgScale - imgBias;
              if (c>=trainColumns-1) k++;
              c = 0;
              if (j+1<len && (data[j+1]=='\n' || data[j+1]=='\r')) mark++;
              i = j + mark;
              mark = -1;
          }
          else i = j + 1;
    }
    validSetSize = 0;
    trainSetSize = 0;
    // CREATE A SUBSAMPLED VERSION OF IMAGES
    if (trainColumns==784){
        for (i=0;i<k;i++){
           for (j1=0;j1<14;j1++)
               for (j2=0;j2<14;j2++){
                   trainImages2[i][14*j1+j2] = (trainImages[i][28*j1*2+j2*2]
                       + trainImages[i][28*j1*2+j2*2+1]
                       + trainImages[i][28*(j1*2+1)+j2*2] 
                       + trainImages[i][28*(j1*2+1)+j2*2+1])/4.0;  
               }
        }
    }
    // CREATE TRAIN AND VALIDATION SETS
    for (i=0;i<k;i++){
       rnd = (float)rand()/(float)RAND_MAX;
       if (rnd<=testProp) validSet[validSetSize++] = i;
       else trainSet[trainSetSize++] = i;
    }
    trainSizeI = k;
    trainSizeE = k;
    free(data);
    return k;
}

/**********************************************************************/
/*      LOAD DATA                                                     */
/**********************************************************************/
int loadTest(int ct, int sh, int rc, float imgScale, float imgBias){
    // LOAD TEST DATA FROM FILE
    if (ct<=0) ct=1e6;
    int i,len = 0, lines=0, lines2=0;;
    float rnd;
    // READ IN TEST.CSV
    char buffer[1000000];
    char name[80] = "test.csv";
    strcpy(name,spGet("dataFile"));
    if (access(name,F_OK)!=0) sprintf(name,"../%s",spGet("dataFile"));
    if (access(name,F_OK)==0){
        data = (char*)malloc((int)fsize(name)+1);
        FILE *fp;
        fp = fopen(name,"r");
        while (fgets(buffer, 1000000, fp)){
            len += sprintf(data+len,"%s",buffer);
            //lines++;
        }
        fclose(fp);
    }
    else {
        sprintf(buffer,"ERROR: File %s not found.",name);
        webwriteline(buffer);
        return 0;
    }
    // COUNT LINES
    for (i=0;i<len;i++){
        if (data[i]=='\n') lines++;
        if (data[i]=='\r') lines2++;
    }
    if (lines2>lines) lines=lines2;
    
    // ALLOCATE MEMORY
    if (testImages!=NULL){
        free(testImages);
        free(testImages2);
        free(testDigits);
    }
    testImages = malloc(784 * lines * sizeof(float));
    testImages2 = malloc(196 * lines * sizeof(float));
    testDigits = malloc(lines * sizeof(int));
    // DECODE COMMA SEPARATED ROWS
    int j = 0, k = 0, c = 0, mark = 0;
    int d = 0, j1,j2;
    while (data[j]!='\n' && data[j]!='\r'){
        if (data[j]==',') c++;
        j++;
    }
    if (data[j]!='\n' || data[j]!='\r') j++;
    testColumns = c+1;
    if (rc==1) {
        testColumns--;
        mark = -1;
    }
    //printf("len=%d lines=%d columns=%d\n",len,lines,testColumns);
    c = 0; i = 0;
    if (sh==1) i = j+1;
    while(i<len && k<ct){
    	j = i; while (data[j]!=',' && data[j]!='\r' && data[j]!='\n') j++;
    	if (data[j]=='\n' || data[j]=='\r') mark = 1;
        data[j] = 0;
    	d = atof(data+i);
        if (mark==-1){
            mark = 0;
        }
        else if (mark==0) {
            testImages[k][c] = d/imgScale - imgBias;
            c++;
        }
        if (mark>=1){
            testImages[k][c] = d/imgScale - imgBias;
            if (c>=testColumns-1) k++;
    	    c = 0;
            if (j+1<len && (data[j+1]=='\n' || data[j+1]=='\r')) mark++;
   	        i = j + mark;
            mark = 0;
            if (rc==1) mark = -1;
        }
        else i = j + 1;
    }
    // CREATE A SUBSAMPLED VERSION OF IMAGES
    if (testColumns==784){
        for (i=0;i<k;i++){
           for (j1=0;j1<14;j1++)
               for (j2=0;j2<14;j2++){
                   testImages2[i][14*j1+j2] = (testImages[i][28*j1*2+j2*2]
                       + testImages[i][28*j1*2+j2*2+1]
                       + testImages[i][28*(j1*2+1)+j2*2]
                       + testImages[i][28*(j1*2+1)+j2*2+1])/4.0;
               }
        }
    }
    testSizeI = k;
    free(data);
    return k;
}

/**********************************************************************/
/*      SPECIAL AND/OR DEBUGGING                                      */
/**********************************************************************/
void dream(int x, int y, int it, float bs, int ds){
    // MAKE A NEURAL NETWORK DREAM
    showCon = 0;
    int i, in = 10-numLayers;
    float (*tImage)[layerSizes[in]];
    if (layerSizes[in]==784) tImage = trainImages;
    else tImage = trainImages2;
    for (i=0;i<layerSizes[in];i++) {
        if (x>=0 && x<=trainSizeI+extraTrainSizeI) tImage[trainSizeE][i] = tImage[x][i];
        else tImage[trainSizeE][i] = 0.1;
    }
    dreamProp(y,it,bs,ds);
    webwriteline("Done");
}

/**********************************************************************/
/*      DISPLAY PROGRESS                                              */
/**********************************************************************/
void displayCDigits(int x, int y){
    // DISPLAY IMAGES FOR SPECIFIC CONFUSION MATRIX CELL
    char buffer[1024];
    int i,j,ct = 0;
    for (i=0;i<maxCD;i++) if (cDigits[x][y][i]==-1) ct++;
    ct = maxCD - ct;
    sprintf(buffer,"Displaying true=%d, predict=%d",x,y);
    if (ct==0) sprintf(buffer,"None to display: true=%d, predict=%d",x,y);
    webwriteline(buffer);
    displayDigits(cDigits[x][y],ct,4,1,0,0,2);
}

/**********************************************************************/
/*      DISPLAY DATA                                                  */
/**********************************************************************/
void displayDigits(int *dgs, int ct, int pane, int train, int cfy, int wd, int big){
    // DISPLAY MULTIPLE IMAGES
    if (ct==0) return;
    int i, j, w, m, col, row, in = 10-numLayers;
    int size = layerSizes[in],st=1,sk=1,sp=7;
    if (size!=196 && size!=784) size=196;
    if (big==1) size=784;
    else if (big==0) size=196;
    if (wd!=0) size=784;
    float (*tImage)[size];
    if (size==784) {tImage = trainImages; w=28; m=1;}
    else {tImage = trainImages2; w=14; m=2;}
    if (train==0 && size==784) tImage = testImages;
    else if (train==0) tImage = testImages2;
    if (wd!=0) {
        w = wd;
        if (w<=14) m=2;
    }
    
    websetcolors(256,red2,green2,blue2,pane);
    for (int i=0;i<240000;i++) ((int*)image)[i] = 128;
    int cc, pp, ss, dt;
    if (ct<=6) {cc = 3; pp = 5; ss = 200;}
    else if (ct<=12) {cc = 4; pp = 3; ss = 133;}
    else if (ct<=24) {cc = 6; pp = 5; ss = 100; m = 1;}
    else if (ct<=35) {cc = 7; pp = 4; ss = 80; m = 1;}
    else {cc = 9; pp = 3; ss = 67; m = 1;}
    if (size==784 && ct>12 && ct<=24) {sk=0; pp=3;}
    else if (size==784 && ct>=24) {sk=0; pp=2;}
    //else if (size==784 && ct>=24) st=2;
    if (ct>35) sp=2;
    showDig[pane-3][0] = ct;
    for (i=0;i<ct;i++){    
        col = i%cc;
        row = i/cc;
        if (dgs[i]!=-1){
            printDigit(row*ss,col*ss,(int*)image,400,600,tImage[dgs[i]],w,w,st,pp*m,sk*m);
            dt = -1;
            if (train==1) dt = trainDigits[dgs[i]];
            int nn = ipGet("NN");
            //nn = 1;
            if (nn==1 && cfy==1 && layers[0]!=NULL) dt = forwardProp(dgs[i],0,train);
            else if (nn==0 && cfy==1) dt = singleKNN(dgs[i],ipGet("k"),ipGet("norm"),3,train,big,0);
            if ((train!=0 || (cfy!=0 && nn==0) || (cfy!=0 && layers[0]!=NULL)) && dt!=-1)
                printInt(row*ss+w*(pp+1+(sk-1))*m/st,col*ss+w*(pp+1+(sk-1))*m/st-7,(int*)image,400,600,dt,255,2);
            printInt(row*ss+w*(pp+1+(sk-1))*m/st,col*ss+sp,(int*)image,400,600,dgs[i],0,2);
        }
        showDig[pane-3][i+1] = dgs[i];
    }
    for (i=0;i<10;i++)
    for (j=0;j<10;j++){
        if (i==j || i+j==9) image[390+i][590+j] = 0; 
        else image[390+i][590+j] = 255;
    }
    webimagedisplay(600,400,(int*)image,pane);
    if (pane==4) showEnt = 0;
    else if (pane==5) showAcc = 0;
    else if (pane==3) showCon = 0;
}

/**********************************************************************/
/*      SPECIAL AND/OR DEBUGGING                                      */
/**********************************************************************/
void dreamProp(int y, int it, float bs, int ds){
    // MAKE A NEURAL NETWORK DREAM
    // NEEDS TO BE UPDATED FOR CONV AND POOL LAYERS
    char buffer[80];
    int i = 0, j, k, n, r = 0, d=0, c;
    int dc, a, a2, i2, j2, i3, j3, pmax, imax, jmax;
    int in = 10-numLayers, row, col, ct=0;
    float der = 1.0, min, max;
    float (*tImage)[layerSizes[in]];
    if (layerSizes[in]==784) tImage = trainImages;
    else tImage = trainImages2;
    for (n=0;n<it;n++){
        c = forwardProp(trainSizeE+n,0,1);
        trainDigits[trainSizeE+n] = c;
        // OUTPUT LAYER - CALCULATE ERRORS
        for (i=0;i<layerSizes[9];i++){
            errors[9][i] = an * (0 - layers[9][i]);
            if (i==y) errors[9][i] = an * (1  - layers[9][i]);
        }
        
        int temp, temp2;
        // HIDDEN LAYERS - CALCULATE ERRORS
        for (k=8;k>9-numLayers;k--){
            if (layerType[k+1]==0) // FEEDS INTO FULLY CONNECTED
            for (i=0;i<layerSizes[k]*layerChan[k];i++){
                errors[k][i] = 0.0;
                if (activation==2) der = (layers[k][i]+1)*(1-layers[k][i]); //TanH derivative
                if (activation==2 || layers[k][i]>0){ //this is ReLU derivative
                    temp = layerSizes[k]*layerChan[k]+1;
                    for (j=0;j<layerSizes[k+1];j++)
                        errors[k][i] += errors[k+1][j]*weights[k+1][j*temp+i]*der;
                }
            }
            else if (layerType[k+1]==1){ // FEEDS INTO CONVOLUTION
                for (i=0;i<layerSizes[k]*layerChan[k];i++) errors[k][i] = 0.0;
                dc = 0; if (layerPad[k+1]==1) dc = layerConv[k+1]/2;
                for (a=0;a<layerChan[k+1];a++)
                for (i=0;i<layerWidth[k+1];i++)
                for (j=0;j<layerWidth[k+1];j++){
                    temp = a*layerSizes[k+1] + i*layerWidth[k+1] + j;
                    temp2 = a*(layerConvStep[k+1]+1);
                    for (a2=0;a2<layerChan[k];a2++)
                    for (i2=0;i2<layerConv[k+1];i2++)
                    for (j2=0;j2<layerConv[k+1];j2++){
                        i3 = i + i2 - dc;
                        j3 = j + j2 - dc;
                        if (activation==2) der = (layers[k][a2*layerSizes[k] + i3*layerWidth[k] + j3]+1)*(1-layers[k][a2*layerSizes[k] + i3*layerWidth[k] + j3]); //TanH
                        if (activation==2 || layers[k][a2*layerSizes[k] + i3*layerWidth[k] + j3]>0) // this is ReLU derivative
                        if (i3>=0 && i3<layerWidth[k] && j3>=0 && j3<layerWidth[k]) // padding
                        errors[k][a2*layerSizes[k] + i3*layerWidth[k] + j3] +=
                            weights[k+1][temp2 + a2*layerConvStep2[k+1] + i2*layerConv[k+1] +j2]
                            * errors[k+1][temp];
                    }
                }
            }
            else if (layerType[k+1]==2){ // FEEDS INTO MAX POOLING
                for (i=0;i<layerSizes[k]*layerChan[k];i++) errors[k][i] = 0.0;
                for (a=0;a<layerChan[k];a++)
                for (i=0;i<layerWidth[k+1];i++)
                for (j=0;j<layerWidth[k+1];j++){
                    pmax = -1e6;
                    for (i2=0;i2<layerConv[k+1];i2++)
                    for (j2=0;j2<layerConv[k+1];j2++)
                        if (layers[k][a*layerSizes[k] + (i*layerConv[k+1]+i2)*layerWidth[k] + j*layerConv[k+1]+j2]>pmax){
                            pmax = layers[k][a*layerSizes[k] + (i*layerConv[k+1]+i2)*layerWidth[k] + j*layerConv[k+1]+j2];
                            imax = i2;
                            jmax = j2;
                        }
                    errors[k][a*layerSizes[k] + (i*layerConv[k+1]+imax)*layerWidth[k] + j*layerConv[k+1]+jmax] =
                        errors[k+1][a*layerSizes[k+1] + i*layerWidth[k+1] + j];
                }
            }
        }
        //for (i=0;i<layerSizes[in];i++) printf("%f ",errors[in][i]);
        //printf("\n");
        float nei;
        int ct;
        for (i=0;i<layerSizes[in];i++)
            tImage[trainSizeE+n+1][i] = tImage[trainSizeE+n][i];
        for (i=0;i<layerSizes[in];i++){
            nei = 0.0; ct = 0;
            if (i-1>=0) {nei += tImage[trainSizeE+n+1][i-1]; ct++;}
            if (i+1<28) {nei += tImage[trainSizeE+n+1][i+1]; ct++;}
            if (i-28>=0) {nei += tImage[trainSizeE+n+1][i-28]; ct++;}
            if (i+28<784) {nei += tImage[trainSizeE+n+1][i+28]; ct++;}
            nei = (nei/ct - tImage[trainSizeE+n+1][i])/bs;
            tImage[trainSizeE+n+1][i] += errors[in][i] + nei;
            if (tImage[trainSizeE+n+1][i]<0.0) tImage[trainSizeE+n+1][i]=0.0;
            if (tImage[trainSizeE+n+1][i]>1.0) tImage[trainSizeE+n+1][i]=1.0;
        }
    }
    c = forwardProp(trainSizeE+it,0,1);
    trainDigits[trainSizeE+it] = c;
    int adj = 0;
    if (it%ds==0) adj=-1;
    int dgs[it/ds+2];
    for (i=0;i<it/ds+2;i++) dgs[i] = trainSizeE+i*ds;
    dgs[it/ds+1] = trainSizeE+it;
    displayDigits(dgs, it/ds+2+adj, 3,1,0,0,2);
}


// THE FOLLOWING ROUTINES ARE FROM WEBGUI'S FILE EXAMPLE.C
/***************************************************************************/
/* WEBGUI A web browser based graphical user interface                     */
/* Version: 1.0 - June 25 2017                                             */
/***************************************************************************/
/* Author: Christopher Deotte                                              */
/* Advisor: Randolph E. Bank                                               */
/* Funding: NSF DMS-1345013                                                */
/* Documentation: http://ccom.ucsd.edu/~cdeotte/webgui                     */
/***************************************************************************/

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void initParameterMap(char* str, int n){
    /* reads array of strings and initializes ip, rp, sp */
    /* and creates a map for accessing ip, rp, and sp */
    int i, index=0;
    for (i=0; i<n; i++) if (str[80*i]=='n') ct++;
    map_keys = (char**)malloc(ct * sizeof(char*));
    map_indices = (int*)malloc(ct * sizeof(int));
    map_array = (char*)malloc(ct * sizeof(char));
    rp_default = (double*)malloc(ct * sizeof(double));
    ip_default = (int*)malloc(ct * sizeof(int));
    sp_default = (char*)malloc(ct * sizeof(char*) * 80);
    rp = (double*)malloc(ct * sizeof(double));
    ip = (int*)malloc(ct * sizeof(int));
    sp = (char*)malloc(ct * sizeof(char*) * 80);
    for (i=0; i<ct; i++) map_keys[i] = (char*)malloc(20 * sizeof(char));
    for (i=0; i<n; i++)
    if (str[80*i]=='n'){
        strcpy(map_keys[index],extractVal(str+80*i,'n'));
        map_indices[index] = atoi(extractVal(str+80*i,'i'))-1;
        map_array[index] = *extractVal(str+80*i,'t');
        if (map_array[index]=='r'){
            rp_default[map_indices[index]] = atof(extractVal(str+80*i,'d'));
            rp[map_indices[index]] = rp_default[map_indices[index]];
        }
        else if (map_array[index]=='i'){
            ip_default[map_indices[index]] = atoi(extractVal(str+80*i,'d'));
            ip[map_indices[index]] = ip_default[map_indices[index]];
        }
        else if (map_array[index]=='s' || map_array[index]=='f' || map_array[index]=='l'){
            strcpy(sp_default+80*map_indices[index],extractVal(str+80*i,'d'));
            strcpy(sp+80*map_indices[index],sp_default+80*map_indices[index]);
        }
        index++;
    }
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
char* extractVal(char* str, char key){
    /* returns the value associated with key in str */
    buffer[0]=0;
    int index1 = 0, index2;
    while (index1<strlen(str)){
        if (str[index1]=='='){
            if (str[index1-1]==key){
                index2 = index1;
                while (index2<strlen(str) && str[index2]!=',') index2++;
                strncpy(buffer,str+index1+1,index2-index1-1);
                buffer[index2-index1-1]=0;
                break;
            }
        }
        index1++;
    }
    return buffer;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
char processCommand(char* str){
    /* returns command char and updates parameters */
    int index1 = 1, index2 = 2;
    while (str[index2]!=' '){
        if (str[index2]==','){
            updateParameter(str,index1,index2);
            index1 = index2;
        }
        index2++;
    }
    if (index2>2) {
        updateParameter(str,index1,index2);
    }
    return str[0];
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void updateParameter(char* str, int index1, int index2){
    /* parses str between index1 and index2 and updates parameter */
    int index3 = index1+1;
    while (str[index3]!='=') index3++;
    str[index2]=0; str[index3]=0;
    char ch = arrayGet(str+index1+1);
    if (ch=='r') rpSet(str+index1+1,atof(str+index3+1));
    else if (ch=='i') ipSet(str+index1+1,atoi(str+index3+1));
    else if (ch=='s' || ch=='f' || ch=='l') spSet(str+index1+1,str+index3+1);
    str[index2]=','; str[index3]='=';
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
char arrayGet(char* key){
    /* returns which array (ip, rp, sp) key belongs to */
    int i;
    char value = ' ';
    for (i=0; i<ct; i++) if (strcmp(map_keys[i],key)==0)
        value = map_array[i];
    return value;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
int ipGet(char* key){
    int i, value = 0;
    for (i=0; i<ct; i++) if (strcmp(map_keys[i],key)==0)
        value = ip[map_indices[i]];
    return value;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void ipSet(char* key, int value){
    int i;
    for (i=0; i<ct; i++) if (strcmp(map_keys[i],key)==0)
        ip[map_indices[i]] = value;
    return;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
double rpGet(char* key){
    int i;
    double value = 0;
    for (i=0; i<ct; i++) if (strcmp(map_keys[i],key)==0)
        value = rp[map_indices[i]];
    return value;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void rpSet(char* key, double value){
    int i;
    for (i=0; i<ct; i++) if (strcmp(map_keys[i],key)==0)
        rp[map_indices[i]] = value;
    return;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
char* spGet(char* key){
    int i;
    buffer[0] = 0;
    for (i=0; i<ct; i++) if (strcmp(map_keys[i],key)==0)
        strcpy(buffer,sp + 80 * map_indices[i]);
    return buffer;
}

/**********************************************************************/
/*        WEBGUI A web browser based graphical user interface         */
/*                                                                    */
/*        Author: Christopher Deotte                                  */
/*                                                                    */
/*        Version: 1.0 - June 25, 2017                                */
/**********************************************************************/
void spSet(char* key, char* value){
    int i;
    for (i=0; i<ct; i++) if (strcmp(map_keys[i],key)==0)
        strcpy(sp + 80 * map_indices[i],value);
    return;
}

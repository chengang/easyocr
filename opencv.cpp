#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cv.h"
#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <ml.h>
#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/socket.h>
#include <error.h>
#include <netinet/in.h>
#include <arpa/inet.h>
using namespace cv;
using namespace std;

int otsu(IplImage *image) {
  assert(NULL != image);

  int width = image->width;
  int height = image->height;
  int x = 0, y = 0;
  int pixelCount[256];
  float pixelPro[256];
  int i, j, pixelSum = width * height, threshold = 0;

  uchar *data = (uchar *)image->imageData;

  //初始化
  for (i = 0; i < 256; i++) {
    pixelCount[i] = 0;
    pixelPro[i] = 0;
  }

  //统计灰度级中每个像素在整幅图像中的个数
  for (i = y; i < height; i++) {
    for (j = x; j < width; j++) {
      pixelCount[data[i * image->widthStep + j]]++;
    }
  }
  //计算每个像素在整幅图像中的比例
  for (i = 0; i < 256; i++) {
    pixelPro[i] = (float)(pixelCount[i]) / (float)(pixelSum);
  }

  //经典ostu算法,得到前景和背景的分割
  //遍历灰度级[0,255],计算出方差最大的灰度值,为最佳阈值
  float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax = 0;
  for (i = 0; i < 256; i++) {
    w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;

    for (j = 0; j < 256; j++) {
      if (j <= i) //背景部分
      {
        //以i为阈值分类，第一类总的概率
        w0 += pixelPro[j];
        u0tmp += j * pixelPro[j];
      } else //前景部分
      {
        //以i为阈值分类，第二类总的概率
        w1 += pixelPro[j];
        u1tmp += j * pixelPro[j];
      }
    }

    u0 = u0tmp / w0;   //第一类的平均灰度
    u1 = u1tmp / w1;   //第二类的平均灰度
    u = u0tmp + u1tmp; //整幅图像的平均灰度
    //计算类间方差
    deltaTmp = w0 * (u0 - u) * (u0 - u) + w1 * (u1 - u) * (u1 - u);
    //找出最大类间方差以及对应的阈值
    if (deltaTmp > deltaMax) {
      deltaMax = deltaTmp;
      threshold = i;
    }
  }
  //返回最佳阈值;
  return threshold;
}
//分段二值化（k为分几段）
void binary3(IplImage *src, IplImage *binimg, int k) {
  char file_dst[200];
  IplImage *hui = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
  // IplImage* binimg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
  cvCvtColor(src, hui, CV_BGR2GRAY); // CV_GAUSSIAN
  float n;                           //代表分块
  int w, h;
  w = src->width;
  h = src->height;
  n = float(w / k) + 0.5;
  for (int i = 0; i < k; i++) {
    cvSetImageROI(hui, cvRect(n * i, 0, n, src->height));
    IplImage *child_img = cvCreateImage(cvGetSize(hui), 8, 1);
    cvCopy(hui, child_img);
    cvResetImageROI(hui);
    //开始二值化
    int thr = otsu(child_img);
    for (int y = 0; y < h; y++) {
      int ww;
      if ((i + 1) * n > w) {
        ww = w;
      } else {
        ww = (i + 1) * n;
      }

      for (int x = n * i; x < ww; x++) {
        if (CV_IMAGE_ELEM(hui, uchar, y, x) > thr) {
          CV_IMAGE_ELEM(binimg, uchar, y, x) = 0;
        } else
          CV_IMAGE_ELEM(binimg, uchar, y, x) = 255;
      }
    }
    cvReleaseImage(&child_img);
  }
  //保存
  // sprintf(file_dst,"E:\\Vin码测试代码\\测试结果\\2\\%s%d.jpg",imgName,k);//cvShowImage("ffff",img_Clone1);
  // cvSaveImage(file_dst,binimg);
  cvReleaseImage(&hui);
  // cvReleaseImage(&binimg);
}
//图像相加
void plusimg(IplImage *img, IplImage *dst) {
  for (int i = 0; i < img->height; i++) {
    for (int j = 0; j < img->width; j++) {
      if (CV_IMAGE_ELEM(img, uchar, i, j) > 100) {
        CV_IMAGE_ELEM(dst, uchar, i, j) = 255;
      }
    }
  }
}
//图片旋转:type=-1时，逆时针旋转90度，否则，顺时针旋转
// clockwise 为true则顺时针旋转，否则为逆时针旋转
IplImage *rotateImage(IplImage *src, int angle, bool clockwise) {
  angle = abs(angle) % 180;
  if (angle > 90) {
    angle = 90 - (angle % 90);
  }
  int width = (double)(src->height * sin(angle * CV_PI / 180.0)) +
              (double)(src->width * cos(angle * CV_PI / 180.0)) - 1; //+ 1
  int height = (double)(src->height * cos(angle * CV_PI / 180.0)) +
               (double)(src->width * sin(angle * CV_PI / 180.0)) - 1; //+ 1
  int tempLength =
      sqrt((double)src->width * src->width + src->height * src->height) + 10; //
  int tempX = (tempLength + 1) / 2 - src->width / 2;
  int tempY = (tempLength + 1) / 2 - src->height / 2;
  int flag = -1;
  IplImage *dst = NULL;
  dst = cvCreateImage(cvSize(width, height), src->depth, src->nChannels);
  cvZero(dst);
  IplImage *temp =
      cvCreateImage(cvSize(tempLength, tempLength), src->depth, src->nChannels);
  cvZero(temp);

  cvSetImageROI(temp, cvRect(tempX, tempY, src->width, src->height));
  cvCopy(src, temp, NULL);
  cvResetImageROI(temp);

  if (clockwise)
    flag = 1;

  float m[6];
  int w = temp->width;
  int h = temp->height;
  m[0] = (float)cos(flag * angle * CV_PI / 180.);
  m[1] = (float)sin(flag * angle * CV_PI / 180.);
  m[3] = -m[1];
  m[4] = m[0];
  // 将旋转中心移至图像中间
  m[2] = w * 0.5f;
  m[5] = h * 0.5f;
  //
  CvMat M = cvMat(2, 3, CV_32F, m);
  cvGetQuadrangleSubPix(temp, dst, &M);

  cvReleaseImage(&temp);
  return dst;
}
//删除不符合要求的连同区域函数
void DeleteMinarea0(IplImage *img) {
  CvSeq *contour = NULL;
  CvMemStorage *storage = cvCreateMemStorage(0);

  // IplImage * Clone_img=cvCreateImage(cvGetSize(img),8,1);
  IplImage *min_img = cvCreateImage(cvGetSize(img), 8, 1);
  cvCopy(img, min_img);
  // cvZero(min_img);
  cvFindContours(min_img, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP,
                 CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  cvReleaseImage(&min_img);
  while (contour) {
    float tmparea = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
    CvRect rect = ((CvContour *)contour)->rect;
    CvPoint p1, p2;
    p1.x = rect.x;
    p1.y = rect.y;
    p2.x = rect.x + rect.width;
    p2.y = rect.y + rect.height;
    //长宽比
    // double atio;
    // atio=(double)rect.width/rect.height;
    //删除面积、高宽、长宽比、离图边距离近的区域
    if (rect.height > 50 || tmparea < 3 || p1.x < 3 || p1.y < 3 ||
        p2.x > img->width - 3 || p2.y > img->height - 3) {
      cvDrawContours(img, contour, cvScalarAll(0), cvScalarAll(0), 0, CV_FILLED,
                     8);
    }
    contour = contour->h_next;
  }
  cvReleaseMemStorage(&storage);
}
void DeleteMinarea(IplImage *img) {
  CvSeq *contour = NULL;
  CvMemStorage *storage = cvCreateMemStorage(0);

  // IplImage * Clone_img=cvCreateImage(cvGetSize(img),8,1);
  IplImage *min_img = cvCreateImage(cvGetSize(img), 8, 1);
  cvCopy(img, min_img);
  // cvZero(min_img);
  cvFindContours(min_img, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP,
                 CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  cvReleaseImage(&min_img);
  while (contour) {
    float tmparea = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
    CvRect rect = ((CvContour *)contour)->rect;
    CvPoint p1, p2;
    p1.x = rect.x;
    p1.y = rect.y;
    p2.x = rect.x + rect.width;
    p2.y = rect.y + rect.height;
    //长宽比
    // double atio;
    // atio=(double)rect.width/rect.height;
    //删除面积、高宽、长宽比、离图边距离近的区域
    if (rect.height > 60 || tmparea < 3 || p1.x < 3 || p1.y < 3 ||
        p2.x > img->width - 3 || p2.y > img->height - 3) {
      cvDrawContours(img, contour, cvScalarAll(0), cvScalarAll(0), 0, CV_FILLED,
                     8);
    }
    contour = contour->h_next;
  }
  cvReleaseMemStorage(&storage);
}
//水平边缘用__第二次识别用
void DeleteMinarea2(IplImage *img) {
  CvSeq *contour = NULL;
  CvMemStorage *storage = cvCreateMemStorage(0);

  // IplImage * Clone_img=cvCreateImage(cvGetSize(img),8,1);
  IplImage *min_img = cvCreateImage(cvGetSize(img), 8, 1);
  cvCopy(img, min_img);
  // cvZero(min_img);
  for (int j = 0; j < img->height; j++) {
    cvSet2D(min_img, j, 0, cvScalar(0));
    cvSet2D(min_img, j, img->width - 1, cvScalar(0));
  }
  for (int i = 0; i < img->width; i++) {
    cvSet2D(min_img, 0, i, cvScalar(0));
    cvSet2D(min_img, img->height - 1, i, cvScalar(0));
  }
  cvFindContours(min_img, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP,
                 CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  cvReleaseImage(&min_img);
  while (contour) {
    float tmparea = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
    CvRect rect = ((CvContour *)contour)->rect;
    CvPoint p1, p2;
    p1.x = rect.x;
    p1.y = rect.y;
    p2.x = rect.x + rect.width;
    p2.y = rect.y + rect.height;
    //长宽比
    // double atio;
    // atio=(double)rect.width/rect.height;
    //删除面积、高宽、长宽比、离图边距离近的区域
    if (rect.width > 20 || rect.height > 180 || rect.height < 30 ||
        tmparea < 10) {
      cvDrawContours(img, contour, cvScalarAll(0), cvScalarAll(0), 0, CV_FILLED,
                     8);
    }
    contour = contour->h_next;
  }
  cvReleaseMemStorage(&storage);
}
void DeleteMinarea1(IplImage *img) {
  CvSeq *contour = NULL;
  CvMemStorage *storage = cvCreateMemStorage(0);

  // IplImage * Clone_img=cvCreateImage(cvGetSize(img),8,1);
  IplImage *min_img = cvCreateImage(cvGetSize(img), 8, 1);
  cvCopy(img, min_img);
  // cvZero(min_img);
  for (int j = 0; j < img->height; j++) {
    cvSet2D(min_img, j, 0, cvScalar(0));
    cvSet2D(min_img, j, img->width - 1, cvScalar(0));
  }
  for (int i = 0; i < img->width; i++) {
    cvSet2D(min_img, 0, i, cvScalar(0));
    cvSet2D(min_img, img->height - 1, i, cvScalar(0));
  }
  cvFindContours(min_img, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP,
                 CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  cvReleaseImage(&min_img);
  while (contour) {
    float tmparea = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
    CvRect rect = ((CvContour *)contour)->rect;
    CvPoint p1, p2;
    p1.x = rect.x;
    p1.y = rect.y;
    p2.x = rect.x + rect.width;
    p2.y = rect.y + rect.height;
    //长宽比
    // double atio;
    // atio=(double)rect.width/rect.height;
    //删除面积、高宽、长宽比、离图边距离近的区域
    if (rect.width > 200 || rect.width < 20 || rect.height > 100 ||
        tmparea < 10) {
      cvDrawContours(img, contour, cvScalarAll(0), cvScalarAll(0), 0, CV_FILLED,
                     8);
    }
    contour = contour->h_next;
  }
  cvReleaseMemStorage(&storage);
}
//倾斜矫正：img原图,src需要矫正图片,矫正之后的图片
IplImage *fcjiaodu(IplImage *img, IplImage *src, CvRect rect) {
  IplImage *huid = cvCreateImage(cvGetSize(src), 8, 1);
  IplImage *img_xuanzhuan = cvCreateImage(cvGetSize(img), 8, 3);
  cvZero(img_xuanzhuan);
  int i, j;
  int w = huid->width;
  int h = huid->height;
  int nLinebyte = huid->widthStep;
  cvCvtColor(src, huid, CV_BGR2GRAY);
  IplImage *sobel = cvCreateImage(cvGetSize(src), 8, 1);
  cvSobel(huid, sobel, 1, 0, 3);
  // cvShowImage("垂直边缘",sobel);
  IplImage *sobel2 = cvCreateImage(cvGetSize(src), 8, 1);
  cvThreshold(sobel, sobel2, 150, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);
  // cvShowImage("垂直边缘2",sobel2);
  int count = 0;
  double a = 0.0;
  unsigned char pixel;
  double x1 = 0.0, y1 = 0.0;
  for (i = 0; i < h; i++) {
    for (j = 0; j < w; j++) {
      pixel = (unsigned char)sobel2->imageData[i * nLinebyte + j];
      if (pixel == 255) {
        x1 += j;
        count++;
        y1 += i;
      }
    }
  }
  x1 = x1 / (double)count;
  y1 = y1 / (double)count;
  int u1, v1;
  double sum = 0.0;
  double sum1 = 0.0;
  for (i = 0; i < h; i++) {
    for (j = 0; j < w; j++) {
      pixel = (unsigned char)sobel2->imageData[i * nLinebyte + j];
      if (pixel == 255) {
        u1 = j - x1;
        v1 = i - y1;
        sum += u1 * v1;
        sum1 += (u1 * u1 - v1 * v1);
      }
    }
  }
  sum = sum * 2.0;
  a = atan(sum / sum1);
  a = a / 2.0;
  a = a / 3.1415926 * 180.0;
  // IplImage *temp=cvCreateImage(cvGetSize(src),8,1);
  CvPoint2D32f center;
  center.x = float(src->width / 2.0 + 0.5);
  center.y = float(src->height / 2.0 + 0.5);
  //计算二维旋转的仿射变换矩阵
  float m[6];
  CvMat M = cvMat(2, 3, CV_32F, m);
  cv2DRotationMatrix(center, a, 1, &M);
  //变换图像，并用黑色填充其余值
  // cout<<"角度="<<a<<endl;
  rect.y = rect.y - 2;
  rect.x = rect.x + 2;
  if (a < 0) {
    rect.height = rect.height - a * 5 + 5;
    rect.width = rect.width + 9;
  }
  if (a > 0) {
    rect.width = rect.width + a * 10 + 5;
    rect.height = rect.height + 9;
  }
  if (rect.y < 0) {
    rect.y = 0;
  }
  if (rect.x < 0) {
    rect.x = 0;
  }
  if (rect.x + rect.width > img->width - 1) {
    rect.width = img_xuanzhuan->width - rect.x - 1;
  }
  if (rect.y + rect.height > img->height - 1) {
    rect.height = img_xuanzhuan->height - rect.y - 1;
  }
  //存矫正后的图片,并返回
  IplImage *xzz = cvCreateImage(cvSize(rect.width, rect.height), 8, 3);
  cvWarpAffine(img, img_xuanzhuan, &M, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,
               cvScalarAll(255));
  // cvShowImage("222",img_xuanzhuan);
  cvSetImageROI(img_xuanzhuan, rect);
  cvCopy(img_xuanzhuan, xzz);
  // cvShowImage("1111222",xzz);
  // cvReleaseMat(&M);
  cvReleaseImage(&sobel);
  cvReleaseImage(&sobel2);
  cvReleaseImage(&huid);
  cvReleaseImage(&img_xuanzhuan);

  return xzz;
}
//字符分割，根据车牌图片的垂直投影宽度和积累的数值，进行字符分割。
void verProjection_calculate(Mat mat1, int *vArr, int number) {
  IplImage pI_1 = IplImage(mat1);
  CvScalar s1;
  int w = mat1.rows; //(高)
  int h = mat1.cols; //(宽)
  int i, j;

  for (i = 0; i < number; i++) {

    vArr[i] = 0;
  }
  //记录垂直投影点的个数
  for (j = 0; j < h; j++) {
    for (i = 0; i < w; i++) {
      s1 = cvGet2D(&pI_1, i, j);
      // cout<<"xxxxxxxxxxxxx"<<endl;
      if (s1.val[0] > 20) {
        //      cout<<"sdsggfds"<<endl;
        vArr[j] += 1;
      }
    }
  }
}
void verProjection_cut(int *vArr, int width, int *number, int threshold,
                       int a[][2]) {
  // threshold = 2
  //    int **a;
  int i, flag = 0;
  int num = 0;
  // vArr[]：垂直投影点的个数
  // a = (int**)malloc(width / 2 * sizeof(int*));
  for (i = 0; i < width - 1; i++) {

    if ((vArr[i] <= threshold) && (vArr[i + 1] > threshold)) {
      // a[num] = (int* )malloc(2 * sizeof(int));
      a[num][0] = i;
      flag = 1;
    } else if ((vArr[i] > threshold) && (vArr[i + 1] <= threshold) &&
               (flag != 0)) {
      a[num][1] = i;
      num += 1;
      flag = 0;
    }
  }
  *number = num;
  // return a;
}
//对分割后的字符进行高度修剪
//多个字符;b==1则保存
IplImage *charprocess0(IplImage *src, char file_dst[], int b) {
  // char file_dst[100];
  IplImage *hui = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
  // 转为灰度图像
  cvCvtColor(src, hui, CV_BGR2GRAY);
  //创建目标二值图像
  IplImage *binimg =
      cvCreateImage(cvGetSize(hui), IPL_DEPTH_8U, 1); //垂直方向裁剪
  IplImage *binimg1 =
      cvCreateImage(cvGetSize(hui), IPL_DEPTH_8U, 1); //水平方向裁剪
  //加入干扰点
  for (int i = 0; i < hui->width / 5; i++) {
    cvSet2D(hui, 0, i, cvScalar(0));
  }
  //进行二值化
  cvThreshold(hui, binimg1, 0, 255,
              CV_THRESH_BINARY | CV_THRESH_OTSU); //水平方向裁剪
  cvSobel(hui, hui, 1, 0, 3);
  cvThreshold(hui, binimg, 0, 255,
              CV_THRESH_BINARY | CV_THRESH_OTSU); //垂直方向裁剪
  cvNot(binimg1, binimg1);
  // binary3(src,binimg,3);
  //处理边缘
  for (int j = 0; j < binimg->height; j++) {
    cvSet2D(binimg, j, 0, cvScalar(0));
    cvSet2D(binimg, j, binimg->width - 1, cvScalar(0));
    cvSet2D(binimg1, j, 0, cvScalar(0));
    cvSet2D(binimg1, j, binimg->width - 1, cvScalar(0));
  }
  for (int i = 0; i < binimg->width; i++) {
    cvSet2D(binimg, 0, i, cvScalar(0));
    cvSet2D(binimg, binimg->height - 1, i, cvScalar(0));
    cvSet2D(binimg1, 0, i, cvScalar(0));
    cvSet2D(binimg1, binimg->height - 1, i, cvScalar(0));
  }
  // cvShowImage(file_dst,binimg);
  //求切割区域的上下坐标点
  CvSeq *contour = NULL;
  CvMemStorage *storage = cvCreateMemStorage(0); //垂直
  CvSeq *contour1 = NULL;
  CvMemStorage *storage1 = cvCreateMemStorage(0); //水平
  int upymin = binimg->height / 2,
      upyminh = 0; // upymin记录最上边缘点y坐标,upyminh为对应的高度;
  int
  maxH = 0,
  maxY = 0, maxX = 0,
  minX =
      binimg
          ->width; // maxH记录最高的连通区域高度大小,maxY位对应的y坐标,maxW=0为对应的宽度;
  int downy = 0, downh = 0; // downy记录剪切的最低边缘点,（？downh为对应的高度）;
  float arearatio = 0.0; //面积比
  // int flag=0;
  //扫描方式：从下往上，从右往左
  cvFindContours(binimg, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL,
                 CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  cvReleaseImage(&binimg);
  cvFindContours(binimg1, storage1, &contour1, sizeof(CvContour),
                 CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  cvReleaseImage(&binimg1);
  while (contour) {
    CvRect rect = ((CvContour *)contour)->rect;
    contour = contour->h_next;
    if (rect.height < 6) {
      continue;
    }
    int xw = rect.x + rect.width;
    minX = min(rect.x, minX);
    maxX = max(xw, maxX);
  }
  cvReleaseMemStorage(&storage);
  while (contour1) {
    CvRect rect1 = ((CvContour *)contour1)->rect;
    contour1 = contour1->h_next;
    if (rect1.height < 10) {
      continue;
    }
    int yh = rect1.y + rect1.height; //为了使式子简单;maxH=rect1.height;//
                                     //maxH=downy-upymin;
    if (downy == 0) {
      downy = yh;
      upymin = rect1.y;
      continue;
    }
    if (upymin <= yh) {
      upymin = rect1.y;
      downy = max(yh, downy);
    }
    if (upymin > yh) {
      if (rect1.height >= maxH) {
        upymin = rect1.y;
        downy = yh;
      }
    }
  }
  cvReleaseMemStorage(&storage1);
  cvReleaseImage(&hui);
  //求剪切高度宽度大小||downy<binimg->height/3
  int ht = downy - upymin + 1;
  int wt = maxX - minX + 1;
  if (ht < 20 || wt < 8) {
    return (NULL);
  }
  if (minX - 6 >= 0) {
    minX = minX - 6;
    wt = wt + 6;
  }
  cvSetImageROI(src, cvRect(minX, upymin, wt, ht));
  IplImage *img = cvCreateImage(cvSize(wt, ht), 8, 3);
  cvCopy(src, img);

  return (img);
}
//,直接二值化单个字符,返回识别字符序号
int charprocess(IplImage *src, int diji, int *zifuxu, vector<IplImage *> &vinCharImg) {
  // char file_dst[100];
  int w = src->width;
  int h = src->height;
  IplImage *hui = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);
  // 转为灰度图像
  cvCvtColor(src, hui, CV_BGR2GRAY);
  //创建目
  IplImage *binimg =
      cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1); //边缘提取二值化图像，剪切用
  IplImage *binimg1 = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U,
                                    1); //直接二值化图像，排除非字符图片
  // sprintf(file_dst,"E:\\Vin码测试代码\\测试结果\\字符分割2\\%s%d_%d.png",imgName,diji,z1);
  // z1++;
  cvSobel(hui, binimg1, 1, 0, 3); //垂直边缘
  cvSobel(hui, binimg, 0, 1, 3);  //水平边缘
  cvThreshold(binimg, binimg, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
  cvThreshold(binimg1, binimg1, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
  plusimg(binimg1, binimg); //得到边缘提取二值化图像
  cvZero(binimg1);
  //加入干扰点
  for (int i = 0; i < w / 7; i++) {
    cvSet2D(hui, 0, i, cvScalar(0));
  }
  //进行二值化
  cvThreshold(hui, binimg1, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
  cvNot(binimg1, binimg1); //得到直接二值化图像
  // cvSaveImage(file_dst,binimg);//保存
  //处理边缘
  for (int j = 0; j < h; j++) {
    cvSet2D(binimg, j, 0, cvScalar(0));
    cvSet2D(binimg, j, w - 1, cvScalar(0));
    cvSet2D(binimg1, j, 0, cvScalar(0));
    cvSet2D(binimg1, j, w - 1, cvScalar(0));
  }
  for (int i = 0; i < w; i++) {
    cvSet2D(binimg, 0, i, cvScalar(0));
    cvSet2D(binimg, h - 1, i, cvScalar(0));
    cvSet2D(binimg1, 0, i, cvScalar(0));
    cvSet2D(binimg1, h - 1, i, cvScalar(0));
  }
  //求切割区域的上下坐标点
  CvSeq *contour = NULL;
  CvMemStorage *storage = cvCreateMemStorage(0);
  //扫描方式：从下往上，从右往左
  cvFindContours(binimg1, storage, &contour, sizeof(CvContour),
                 CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  cvReleaseImage(&binimg1);
  int upymin = h, upyminh = 0; // upymin记录最上边缘点y坐标,upyminh为对应的高度;
  int
  maxH = 0,
  maxY = 0, maxX = 0,
  maxW =
      0; // maxH记录最高的连通区域高度大小,maxY位对应的y坐标,maxW=0为对应的宽度;
  int downy = 0, downh = 0; // downy记录剪切的最低边缘点,（？downh为对应的高度）;
  while (contour) //开始排除非字符
  {
    CvRect rect = ((CvContour *)contour)->rect;
    contour = contour->h_next;

    if (rect.height < 7) {
      continue;
    }
    if (downy == 0) {
      downy = rect.y + rect.height;
      upymin = rect.y;
      maxH = rect.height;
      maxW = rect.width;
      maxX = rect.x + rect.width;
      continue;
    }
    int yh = rect.y + rect.height; //为了使式子简单
    if (upymin < (yh + 10)) {
      upymin = rect.y;
      downy = max(yh, downy);
      maxW = max(maxW, rect.width);
      maxX = max(rect.x + rect.width, maxX);
      maxH = max(maxH, (downy - rect.y));
    }
    if (upymin >= (yh + 10)) {
      if (rect.height >= maxH) {
        upymin = rect.y;
        downy = max(yh, downy);
        maxX = max(rect.x + rect.width, maxX);
        maxH = rect.height;
      }
    }
  }
  cvReleaseMemStorage(&storage);
  if (maxH < h / 5 || upymin >= h / 2 - 3 || maxW < 6 || maxX <= w / 2 ||
      downy < h / 3 || w < 10) {
    cvReleaseImage(&hui);
    cvReleaseImage(&binimg);
    return (0);
  }
  //开始求字符剪切坐标点
  CvSeq *contour1 = NULL;
  CvMemStorage *storage1 = cvCreateMemStorage(0);

  cvFindContours(binimg, storage1, &contour1, sizeof(CvContour),
                 CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  cvReleaseImage(&binimg);
  downy = 0;
  maxH = 0;
  upymin = h; //从新赋初值
  while (contour1) {
    CvRect rect1 = ((CvContour *)contour1)->rect;
    contour1 = contour1->h_next;

    if (rect1.height < 7) {
      continue;
    }
    if (downy == 0) {
      downy = rect1.y + rect1.height;
      upymin = rect1.y;
      maxH = rect1.height;
      continue;
    }
    int yh1 = rect1.y + rect1.height; //为了使式子简单
    if (upymin < (yh1 + 6)) {
      upymin = rect1.y;
      downy = max(yh1, downy);
      maxH = max(maxH, (downy - rect1.y));
    }
    if (upymin >= (yh1 + 6)) {
      if (rect1.height >= maxH) {
        upymin = rect1.y;
        downy = yh1;
        maxH = rect1.height;
      }
    }
  }
  cvReleaseMemStorage(&storage1);
  cvReleaseImage(&hui);
  //求剪切高度大小||downy<binimg->height/3
  int ht = downy - upymin;
  if (ht < 29) {
    return (0);
  }
  if (ht < 31) {
    ht = h - upymin;
  }
  cvSetImageROI(src, cvRect(0, upymin, w, ht));
  (*zifuxu)++;
  // cv::Mat matimg;
  // matimg=cv::Mat(src);
  // cout<<"sizeqian:"<<vinMatImg.size()<<endl;
  // vinMatImg.push_back(matimg);
  // cout<<"sizehou:"<<vinMatImg.size()<<endl;

  vinCharImg.push_back(src); //保存剪切后的字符
  //加识别程序___________________
  return (0);
}
//字符再分割，垂直边缘//flag=1表示对字符进行一次剪切
void re_cutagain(cv::Mat img, IplImage *pI_1, int threshold, int sw, int diji,
                 int flag, int *zifuxu, vector<IplImage *> &vinCharImg) {
  // char file_dst[100];//保存图片路径用
  cv::Mat img_5; //要分割图片对应的二值化图片
  CvScalar s1;
  CvScalar s2;
  IplImage *pI_3; //存放分割出来的图片
  int width, pic_width;
  int height;
  int i = 0, j = 0, k;
  int pic_ArrNumber;
  int pic_Arr[500][2];
  int vArr[1000]; //存储统计水平投影
  // char str[100];
  // float percentage = 0.0;
  height = img.rows;                        //列高
  width = img.cols;                         //行宽
  verProjection_calculate(img, vArr, 1000); //垂直投影统计
  // pic_Arr = verProjection_cut(vArr,1000,
  // &pic_ArrNumber,threshold);//保存分割数据
  verProjection_cut(vArr, 1000, &pic_ArrNumber, threshold, pic_Arr);
  threshold++;
  int wbiaoji = 0; //标记宽度小于5 的横坐标起始点
  int f1 = 0;

  for (i = 0; i < pic_ArrNumber; i++) {
    pic_width = pic_Arr[i][1] - pic_Arr[i][0];
    // 宽度小于一定值，认为是干扰点，排除___________
    if (pic_width < 6) {
      wbiaoji = pic_Arr[i][0];
      f1 = 1;
      continue;
    }
    //宽度大于一定值，则开始分割出，copy到新的图片pI_3
    int w1 = pic_Arr[i][0]; //起点
    int w2 = pic_Arr[i][1]; //终点
    int ww = pic_Arr[i][1] - pic_Arr[i][0];

    if (f1 == 1) {
      ww = pic_Arr[i][1] - wbiaoji;
      w1 = wbiaoji;
      f1 = 0;
    }
    pI_3 = cvCreateImage(cvSize(ww, height), 8, 3);
    //宽度:在左右多加三个像素点
    if ((pic_Arr[i][1] + 4) < pI_1->width) {
      w2 = w2 + 4;
      pI_3 = cvCreateImage(cvSize(ww + 4, height), 8, 3);
    }
    for (j = 0; j < height; j++) //高度不变
    {
      for (k = w1; k < w2; k++) //宽度:在左右多家三个像素点
      {
        s1 = cvGet2D(pI_1, j, k);
        cvSet2D(pI_3, j, k - w1, s1);
      }
    }

    //————-——————宽度大于一定值则重新分割——————————————————————————————————————————————————————
    if (pic_Arr[i][1] - pic_Arr[i][0] > sw) {
      //保存分割出宽度大于一定值的图片
      if (threshold < 6) {
        IplImage *des2 = cvCreateImage(cvGetSize(pI_3), IPL_DEPTH_8U, 1);
        IplImage *hui = cvCreateImage(cvGetSize(pI_3), IPL_DEPTH_8U, 1);
        cvCvtColor(pI_3, hui, CV_BGR2GRAY); // CV_GAUSSIAN

        int bin = otsu(hui) - threshold * 3;
        cvThreshold(hui, des2, bin, 255, CV_THRESH_BINARY);
        cvNot(des2, des2);
        //保存得到的二值化图片
        //得到二值图片后，进行边缘处理
        for (int j = 0; j < des2->height; j++) {
          cvSet2D(des2, j, 0, cvScalar(0));
          cvSet2D(des2, j, 1, cvScalar(0));
        }
        //开始从新分割_______________
        img_5 = cv::Mat(des2);
        cvReleaseImage(&hui);
        // cvReleaseImage(&des2);

        re_cutagain(img_5, pI_3, threshold, sw, diji, flag, zifuxu, vinCharImg);
        cvReleaseImage(&des2);

      } else //从中间切开
      {
        IplImage *qie1 =
            cvCreateImage(cvSize(pI_3->width / 2, pI_3->height), 8, 3);
        cvSetImageROI(pI_3, cvRect(0, 0, pI_3->width / 2, pI_3->height));
        cvCopy(pI_3, qie1);
        cvResetImageROI(pI_3);
        if (flag == 1) {
          charprocess(qie1, diji, zifuxu, vinCharImg);
        } else {
          (*zifuxu)++;
          // cv::Mat matimg;
          // matimg=cv::Mat(qie1);
          // vinMatImg.push_back(matimg);
          vinCharImg.push_back(qie1);
        }
        int w = (float)pI_3->width / 2.0 + 0.5;
        IplImage *qie2 = cvCreateImage(cvSize(w, pI_3->height), 8, 3);
        cvSetImageROI(pI_3,
                      cvRect(pI_3->width / 2, 0, pI_3->width, pI_3->height));
        cvCopy(pI_3, qie2);
        if (flag == 1) {
          charprocess(qie2, diji, zifuxu, vinCharImg);
        } else {
          (*zifuxu)++;
          /// cv::Mat matimg1;
          // matimg1=cv::Mat(qie2);
          // vinMatImg.push_back(matimg1);

          vinCharImg.push_back(qie2);
        }
      }
      continue;
    }
    //保存分割出来的图片
    if (flag == 1) {
      charprocess(pI_3, diji, zifuxu, vinCharImg);
    } else {
      (*zifuxu)++;
      vinCharImg.push_back(pI_3);
    }
  }
}
//字符分割，flag==1表示首次分割，其他表示首次分割字符数<17后的再次分割
void re_cut(cv::Mat img, IplImage *pI_1, int threshold, int sw, int diji,
            int flag, int *zifuxu, vector<IplImage *> &vinCharImg) {
  CvScalar s1;
  CvScalar s2;
  IplImage *pI_3 = NULL; //存放分割出来的图片
  int pic_width = 0;
  int i = 0, j = 0, k;
  int pic_ArrNumber;
  int pic_Arr[500][2];
  int vArr[1000]; //存储统计水平投影
  // char str[100];
  // float percentage = 0.0;
  // char file_dst[100];//保存图片路径用
  int height = img.rows; //列高
  int width = img.cols;  //行宽
  // cout<<height<<"h_w"<<width<<endl;
  verProjection_calculate(img, vArr, 1000); //垂直投影统计
  // pic_Arr = verProjection_cut(vArr,1000,
  // &pic_ArrNumber,threshold);//保存分割数据
  verProjection_cut(vArr, 1000, &pic_ArrNumber, threshold, pic_Arr);
  threshold++;     //分割阈值
  int wbiaoji = 0; //标记宽度小于5 的横坐标起始点
                   //   cout<<pic_ArrNumber<<endl;
  for (i = 0; i < pic_ArrNumber; i++) {
    //字符宽度
    pic_width = pic_Arr[i][1] - pic_Arr[i][0];
    //    cout<<pic_width<<endl;
    if (pic_width < 6) {
      continue;
    }
    int w1 = pic_Arr[i][0]; //起点
    int w2 = pic_Arr[i][1]; //终点
    //	pI_3=cvCreateImage(cvSize( pic_width,height),8,3);
    //宽度:在左右多加三个像素点
    if ((pic_Arr[i][1] + 4) < pI_1->width) {
      w2 = w2 + 4;
      pI_3 = cvCreateImage(cvSize(pic_width + 4, height), 8, 3);
    } else {
      pI_3 = cvCreateImage(cvSize(pic_width, height), 8, 3);
    }
    /////////////////////////////////
    for (j = 0; j < height; j++) //高度不变
    {
      for (k = w1; k < w2; k++) //宽度:在左右多家三个像素点
      {
        s1 = cvGet2D(pI_1, j, k);
        cvSet2D(pI_3, j, k - w1, s1);
      }
    }

    //————-——————宽度大于一定值则重新分割——————————————
    if (pic_Arr[i][1] - pic_Arr[i][0] > sw) {
      //保存分割出宽度大于一定值的图片
      // sprintf(file_dst,"E:\\Vin码测试代码\\测试结果\\1\\%s%d_%d.jpg",imgName,diji,zifucount);
      // cvSaveImage(file_dst,pI_3);
      IplImage *pic = NULL;
      pic = charprocess0(pI_3, 0, 0);

      if (pic != NULL) {
        //保存
        // sprintf(file_dst,"E:\\Vin码测试代码\\测试结果\\1\\%s%d_%d.png",imgName,diji,zifucount);
        IplImage *des2 = cvCreateImage(cvGetSize(pic), IPL_DEPTH_8U, 1);
        IplImage *hui = cvCreateImage(cvGetSize(pic), IPL_DEPTH_8U, 1);
        cvCvtColor(pic, hui, CV_BGR2GRAY); // CV_GAUSSIAN
        int bin = otsu(hui) - threshold * 2;
        cvThreshold(hui, des2, bin, 255, CV_THRESH_BINARY);
        cvNot(des2, des2);
        cvReleaseImage(&hui);

        //保存得到的二值化图片
        for (int j = 0; j < des2->height; j++) {
          cvSet2D(des2, j, 0, cvScalar(0));
          cvSet2D(des2, j, 1, cvScalar(0));
        }
        //开始从新分割_______________
        // cvSaveImage(file_dst,des2);
        cv::Mat img_5; //要分割图片对应的二值化图片
        img_5 = cv::Mat(des2);
        // cvReleaseImage(&des2);

        re_cutagain(img_5, pic, threshold, sw, diji, flag, zifuxu, vinCharImg);
        cvReleaseImage(&des2);
      }
      cvReleaseImage(&pic);
      continue;
    }
    //保存分割出来的图片
    if (flag == 1) {
      charprocess(pI_3, diji, zifuxu, vinCharImg);
    } else {
      (*zifuxu)++;

      vinCharImg.push_back(pI_3); //保存分
    }
  }
  // cvReleaseImage(&pI_3);
}
//检测训练
void DetectTrainvin() {
  vector<string> img_path; //输入文件名变量
  vector<int> img_catg;
  int nLine = 0;
  string buf;
  ifstream svm_data(
      /*"E:\\测试图片\\yangben\\SVM_DATA.txt"*/ "E:"
                                                "\\Vin码测试代码\\train\\list."
                                                "txt"); //首先，这里搞一个文件列表，把训练样本图片的路径都写在这个txt文件中，使用bat批处理文件可以得到这个txt文件
  unsigned long n;

  while (svm_data) //将训练样本文件依次读取进来
  {
    if (getline(svm_data, buf)) {
      nLine++;
      if (nLine % 2 ==
          0) //这里的分类比较有意思，看得出来上面的SVM_DATA.txt文本中应该是一行是文件路径，接着下一行就是该图片的类别，可以设置为0或者1，当然多个也无所谓
      {
        img_catg.push_back(atoi(
            buf.c_str())); // atoi将字符串转换成整型，标志（0,1），注意这里至少要有两个类别，否则会出错
      } else {
        img_path.push_back(buf); //图像路径
      }
    }
  }
  svm_data.close(); //关闭文件

  CvMat *data_mat, *res_mat;
  int nImgNum =
      nLine / 2; //读入样本数量 ，因为是每隔一行才是图片路径，所以要除以2
  ////样本矩阵，nImgNum：横坐标是样本数量， WIDTH *
  ///HEIGHT：样本特征向量，即图像大小 4752
  data_mat = cvCreateMat(
      nImgNum, 5184,
      CV_32FC1); //这里第二个参数，即矩阵的列是由下面的descriptors的大小决定的，可以由descriptors.size()得到，且对于不同大小的输入训练图片，这个值是不同的
  cvSetZero(data_mat);
  //类型矩阵,存储每个样本的类型标志
  res_mat = cvCreateMat(nImgNum, 1, CV_32FC1);
  cvSetZero(res_mat);

  IplImage *src;
  IplImage *trainImg = cvCreateImage(
      cvSize(232, 28), 8,
      3); //需要分析的图片，这里默认设定图片是64*64大小，所以上面定义了1764，如果要更改图片大小，可以先用debug查看一下descriptors是多少，然后设定好再运行

  //开始搞HOG特征
  for (string::size_type i = 0; i != img_path.size(); i++) {
    src = cvLoadImage(img_path[i].c_str(), 1);
    if (src == NULL) {
      cout << " can not load the image: " << img_path[i].c_str() << endl;
      continue;
    }

    cout << " processing " << img_path[i].c_str() << endl;

    cvResize(
        src,
        trainImg); //读取图片 cv::Size(80,20), cv::Size(16,4), cv::Size(4,2),
                   //cv::Size(4,2), 9
                   //cvSize(232,28),cvSize(32,16),cvSize(20,6),cvSize(8,4),9
                   //__98%
    HOGDescriptor *hog =
        new HOGDescriptor(cvSize(232, 28), cvSize(36, 16), cvSize(28, 6),
                          cvSize(6, 4), 9); //具体意思见参考文章1,2
    vector<float> descriptors;              //结果数组e

    hog->compute(trainImg, descriptors, Size(1, 1),
                 Size(0, 0)); //调用计算函数开始计算
    cout << "HOG dims: " << descriptors.size() << endl;
    //	cvShowImage("PPPPP",trainImg);
    // CvMat* SVMtrainMat=cvCreateMat(descriptors.size(),1,CV_32FC1);
    n = 0;
    for (vector<float>::iterator iter = descriptors.begin();
         iter != descriptors.end(); iter++) {
      cvmSet(data_mat, i, n, *iter); //把HOG存储下来
      n++;
    }
    // cout<<SVMtrainMat->rows<<endl;
    cvmSet(res_mat, i, 0, img_catg[i]);
    cout << " end processing " << img_path[i].c_str() << " " << img_catg[i]
         << endl;
  }

  CvSVM svm;         //新建一个SVM
  CvSVMParams param; //这里是参数
  CvTermCriteria criteria;
  criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
  param = CvSVMParams(CvSVM::C_SVC, CvSVM::LINEAR, 10.0, 0.09, 1.0, 10.0, 0.5,
                      1.0, NULL, criteria);

  //☆☆☆☆☆☆☆☆☆(5)SVM学习☆☆☆☆☆☆☆☆☆☆☆☆
  svm.train(data_mat, res_mat, NULL, NULL, param); //训练啦
  //☆☆利用训练数据和确定的学习参数,进行SVM学习☆☆☆☆
  svm.save("SVM_DATA.xml");

  cvReleaseMat(&data_mat);
  cvReleaseMat(&res_mat);
}
//第一次识别提取src为彩色图像；img为得到连通区域后的二值图像__________________________________________________________________________
bool ExtrVin(IplImage *src, IplImage *img, CvSVM *svmvin, vector<IplImage *> &vinCharImg) {
  CvSeq *contour = NULL;
  CvMemStorage *storage = cvCreateMemStorage(0);
  IplImage *img_Clone = cvCreateImage(cvGetSize(img), 8, 1);
  IplImage *img_Clone1 = cvCreateImage(cvGetSize(src), 8, 3);

  DeleteMinarea(img);
  char file_dst[200];      //保存路径
  cvCopy(src, img_Clone1); //归一化后的彩色图像
  cvCopy(img, img_Clone);  //得到的连通区域图片
  int k = 0;               //保存图片用
  bool det = false;             //标记是否检测出正样本
                           // CV_RETR_EXTERNAL：只检索最外面的轮廓；CV_RETR_LIST：检索所有的轮廓，并将其放入list中；CV_RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边
                           //界，第二层是空洞的边界；
  // CV_RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次。
  cvFindContours(img_Clone, storage, &contour, sizeof(CvContour),
                 CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  //释放
  cvReleaseImage(&img_Clone);
  int diji = 0; //记录每张图片中判定为vin码的个数

  vector<int> everycharNum; //存储每个图像检测到的每个可能Vin区域分割的字符数
  while (contour) {

    CvRect rect = ((CvContour *)contour)->rect;
    float tmparea = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
    CvRect rect1; //获取感兴趣的矩形区域
    rect1.x = rect.x - 3;
    rect1.y = rect.y - 3;
    rect1.width = rect.width + 6;
    rect1.height = rect.height + 6;
    //
    if (tmparea > 800 && tmparea < 20000 && rect.width > 150 &&
        rect.height < 150) {
      cvSetImageROI(img_Clone1, rect1); //获取感兴趣区域
      IplImage *img_c = cvCreateImage(cvSize(rect1.width, rect1.height), 8, 3);
      IplImage *img_cc = NULL;
      IplImage *xuanzhuan = NULL;
      cvCopy(img_Clone1, img_c); //重置感兴趣区域
      cvResetImageROI(img_Clone1);
      xuanzhuan = fcjiaodu(src, img_c, rect);
      cvReleaseImage(&img_c); //释放
      // sprintf(file_dst,"E:\\Vin码测试代码\\测试结果\\矫正\\%s%d.jpg",imgName,k);
      img_cc =
          charprocess0(xuanzhuan, file_dst,
                       0); // 0表示不保存图片，此函数已对xuanzhuan进行了释放//
      cvReleaseImage(&xuanzhuan); //释放

      if (img_cc !=
          NULL) ////开始分类检测____________________________________________________________________________
      {
        IplImage *trainImg = cvCreateImage(cvSize(232, 28), 8, 3);
        cvZero(trainImg);
        //判断是否旋转________________________________

        // if(rect.y<img->height/3&&rect.x<img->width/4)//如果定位的位置在左上角则旋转图片180°
        //{
        //    IplImage* r_img_cc1=NULL;IplImage* r_img_cc2=NULL;
        //  r_img_cc1=rotateImage(img_cc, 90, TRUE);
        // r_img_cc2=rotateImage(r_img_cc1, 90, TRUE);
        // cvResize(r_img_cc2,trainImg);
        // cvReleaseImage(&r_img_cc1);cvReleaseImage(&r_img_cc2);
        // }
        // else
        cvResize(img_cc, trainImg); //归一化大小

        //读取图片     240,40),cvSize(16,8),cvSize(8,4),cvSize(16,8),9
        ////cvSize(232,24),cvSize(32,12),cvSize(20,6),cvSize(8,6),9
        HOGDescriptor *hog =
            new HOGDescriptor(cvSize(232, 28), cvSize(36, 16), cvSize(28, 6),
                              cvSize(6, 4), 9); //具体意思见参考文章1,2
        vector<float> descriptors;              //结果数组
        hog->compute(trainImg, descriptors, Size(1, 1),
                     Size(0, 0)); //调用计算函数开始计算
        cvReleaseImage(&trainImg);
        // cout<<"HOG dims: "<<descriptors.size()<<endl;
        CvMat *SVMtrainMat = cvCreateMat(1, descriptors.size(), CV_32FC1);
        int n = 0;
        for (vector<float>::iterator iter = descriptors.begin();
             iter != descriptors.end(); iter++) {
          cvmSet(SVMtrainMat, 0, n, *iter);
          n++;
        }
        int ret = svmvin->predict(
            SVMtrainMat); //获取最终检测结果，这个predict的用法见 OpenCV的文档
        //若是标志则保存  分类结果红正
        vector<float>(descriptors).swap(descriptors);
        cvReleaseMat(&SVMtrainMat);
        if (ret == 1) //预测为正样本，则进行字符分割
        {
          det = TRUE; //标志定位到了Vin码
                      //尺度变换,得到nimg
                      ////CV_INTER_CUBIC,CV_INTER_NNcvShowImage("SDS",xuan);
          float atio = (float)img_cc->height / img_cc->width;
          IplImage *nimg = cvCreateImage(cvSize(480, 480 * atio), 8, 3);
          //尺寸归一化
          cvResize(img_cc, nimg);
          //二值化，为分割做准备
          IplImage *binaryimg =
              cvCreateImage(cvSize(nimg->width, nimg->height), 8, 1);
          binary3(nimg, binaryimg, 14);
          //字符分割
          int sw = binaryimg->width / 16; //字符超过sw再次进行分割
          int threshold = 1;              //垂直直方图统计分割阈值
          for (int j = 0; j < binaryimg->height; j++) {
            cvSet2D(binaryimg, j, 0, cvScalar(0));
          }
          //开始字符分割
          cv::Mat matimg;
          matimg = cv::Mat(binaryimg);
          //zifucount = 0; //分割出字符数
          int zifuxu = 0;    //对字符剪切后的字符数
          // printf("开始字符分割\n");
          re_cut(matimg, nimg, threshold, sw, ++diji, 1, &zifuxu, vinCharImg);
          //如果分割出的字符数目少于17

          int ffff = 0;
          // cout<<"sizemat:"<<vinMatImg.size()<<endl;
          // cout<<"sizeiplimag:"<<vinCharImg.size()<<endl;
          // cout<<"zifuxu:"<<zifuxu<<endl;
          if (zifuxu < 17) {

            for (int i = vinCharImg.size() - zifuxu; i < vinCharImg.size();
                 i++) {
              cvReleaseImage(&vinCharImg[i]);
            }
            vinCharImg.erase(vinCharImg.end() - zifuxu, vinCharImg.end());
            ffff = 1;
            zifuxu = 0;
            re_cut(matimg, nimg, threshold, sw, diji, 0, &zifuxu, vinCharImg);
          }

          if (zifuxu < 17 && ffff == 1) {
            for (int i = vinCharImg.size() - zifuxu; i < vinCharImg.size();
                 i++) {
              cvReleaseImage(&vinCharImg[i]);
            }
            vinCharImg.erase(vinCharImg.end() - zifuxu, vinCharImg.end());
          }
          if (zifuxu >= 17) {
            if (everycharNum.size() != 0) {
              if (zifuxu > everycharNum[0]) {

                for (int i = vinCharImg.size() - zifuxu; i < vinCharImg.size();
                     i++) {
                  cvReleaseImage(&vinCharImg[i]);
                }
                vinCharImg.erase(vinCharImg.end() - zifuxu, vinCharImg.end());

              } else {
                for (int i = 0; i < everycharNum[0]; i++) {
                  cvReleaseImage(&vinCharImg[i]);
                }
                vinCharImg.erase(vinCharImg.begin(),
                                 vinCharImg.begin() + everycharNum[0]);
                everycharNum.push_back(zifuxu);
              }
            } else {
              everycharNum.push_back(zifuxu);
            }
          }

          cvReleaseImage(&binaryimg);
          cvReleaseImage(&nimg);
        }
      }
      cvReleaseImage(&img_cc);
    }
    k++;
    contour = contour->h_next;
  }
  vector<int>(everycharNum).swap(everycharNum);
  cvReleaseMemStorage(&storage);
  cvReleaseImage(&img_Clone1);
  return det;
}
//第二次识别
bool ExtrVin1(IplImage *src, IplImage *img, CvSVM *svmvin, vector<IplImage *> &vinCharImg) {
  CvSeq *contour = NULL;
  CvMemStorage *storage = cvCreateMemStorage(0);
  IplImage *img_Clone = cvCreateImage(cvGetSize(img), 8, 1);
  IplImage *img_Clone1 = cvCreateImage(cvGetSize(src), 8, 3);
  int w = img->width;
  int h = img->height;
  for (int j = 0; j < h; j++) {
    cvSet2D(img, j, 0, cvScalar(0));
    cvSet2D(img, j, w - 1, cvScalar(0));
  }
  //________
  for (int i = 0; i < w; i++) {
    cvSet2D(img, 0, i, cvScalar(0));
    cvSet2D(img, h - 1, i, cvScalar(0));
  }
  // img=DeleteMinarea(img);
  // cvShowImage("sds",img);
  char file_dst[200];      //保存路径
  cvCopy(src, img_Clone1); //归一化后的彩色图像
  cvCopy(img, img_Clone);  //得到的连通区域图片
  int k = 0;               //保存图片用
  bool det = FALSE;             //标记是否检测出正样本
                           // CV_RETR_EXTERNAL：只检索最外面的轮廓；CV_RETR_LIST：检索所有的轮廓，并将其放入list中；CV_RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边
                           //界，第二层是空洞的边界；
  // CV_RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次。
  cvFindContours(img_Clone, storage, &contour, sizeof(CvContour),
                 CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  cvReleaseImage(&img_Clone);
  int diji = 0; //记录张图片中判定为vin码的个数

  vector<int> everycharNum; //存储每个图像检测到的每个可能Vin区域分割的字符数
  while (contour) {

    CvRect rect = ((CvContour *)contour)->rect;
    float tmparea = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
    CvRect rect1;
    ;
    float bilv = (float)rect.width / rect.height;

    rect1.x = rect.x; // if(rect1.x<=0){rect1.x=rect.x;}
    rect1.width = rect.width + 6;
    if (rect1.width + rect1.x > w - 1) {
      rect1.width = rect.width;
    }
    rect1.y = rect.y + rect.height - 43;
    rect1.height = 43;
    if (rect.width > 200 && rect.width < 250) {
      rect1.y = rect.y + rect.height - 38;
      rect1.height = 38;
    }
    if (rect.width > 180 && rect.width < 201) {
      rect1.y = rect.y + rect.height - 28;
      rect1.height = 28;
    }
    if (rect.width < 181) {
      rect1.y = rect.y + rect.height - 23;
      rect1.height = 23;
    }
    if (rect1.y < 0) {
      rect1.y = 0;
    }
    if (rect1.y + rect1.height > h - 1) {
      rect1.height = h - rect1.y - 1;
    }
    // cvRectangle(src,p1,p2,CV_RGB(0,0,255),1,8,0);
    //改一下阈值原来是：tmparea>800&&tmparea<20000&&rect.width>150&&rect.height<150
    if (rect.width > 100 && rect.width < w * 3 / 4) {
      cvSetImageROI(img_Clone1, rect1); //获取感兴趣区域
      IplImage *img_c = cvCreateImage(cvSize(rect1.width, rect1.height), 8, 3);
      cvCopy(img_Clone1, img_c);
      IplImage *img_cc = 0;
      IplImage *xuanzhuan = 0;
      xuanzhuan = fcjiaodu(src, img_c, rect1);
      cvReleaseImage(&img_c); //释放
      img_cc =
          charprocess0(xuanzhuan, file_dst,
                       0); // 0表示不保存图片，此函数已对xuanzhuan进行了释放//
      cvReleaseImage(&xuanzhuan); //释放
      if (img_cc !=
          NULL) ////开始分类检测____________________________________________________________________________
      {
        IplImage *trainImg = cvCreateImage(cvSize(232, 28), 8, 3);
        cvZero(trainImg);
        cvResize(img_cc, trainImg); //归一化大小
        //读取图片     240,40),cvSize(16,8),cvSize(8,4),cvSize(16,8),9
        ////cvSize(232,24),cvSize(32,12),cvSize(20,6),cvSize(8,6),9
        HOGDescriptor *hog =
            new HOGDescriptor(cvSize(232, 28), cvSize(36, 16), cvSize(28, 6),
                              cvSize(6, 4), 9); //具体意思见参考文章1,2
        vector<float> descriptors;              //结果数组
        hog->compute(trainImg, descriptors, Size(1, 1),
                     Size(0, 0)); //调用计算函数开始计算
        // cout<<"HOG dims: "<<descriptors.size()<<endl;
        CvMat *SVMtrainMat = cvCreateMat(1, descriptors.size(), CV_32FC1);
        int n = 0;
        for (vector<float>::iterator iter = descriptors.begin();
             iter != descriptors.end(); iter++) {
          cvmSet(SVMtrainMat, 0, n, *iter);
          n++;
        }
        int ret = svmvin->predict(
            SVMtrainMat); //获取最终检测结果，这个predict的用法见 OpenCV的文档
        //若是标志则保存  分类结果红正
        cvReleaseImage(&trainImg);
        cvReleaseMat(&SVMtrainMat);
        vector<float>(descriptors).swap(descriptors);

        if (ret == 1) //预测为正样本，则进行字符分割
        {
          det = TRUE;
          float atio = (float)img_cc->height / img_cc->width;
          IplImage *nimg = cvCreateImage(cvSize(480, 480 * atio), 8, 3);
          cvResize(img_cc, nimg);
          //二值化，为分割做准备
          IplImage *binaryimg =
              cvCreateImage(cvSize(nimg->width, nimg->height), 8, 1);
          binary3(nimg, binaryimg, 14);
          //字符分割
          int sw = binaryimg->width / 16; //字符超过sw再次进行分割
          int threshold = 1;              //垂直直方图统计分割阈值
          for (int j = 0; j < binaryimg->height; j++) {
            cvSet2D(binaryimg, j, 0, cvScalar(0));
          }
          //开始字符分割
          cv::Mat matimg;
          matimg = cv::Mat(binaryimg);
          // zifucount=0; //分割出字符数
          int zifuxu = 0; //对字符剪切后的字符数
          re_cut(matimg, nimg, threshold, sw, ++diji, 1, &zifuxu, vinCharImg);
          // cout<<"size:"<<vinCharImg.size()<<endl;
          int ffff = 0;
          if (zifuxu < 17) {

            for (int i = vinCharImg.size() - zifuxu; i < vinCharImg.size();
                 i++) {
              cvReleaseImage(&vinCharImg[i]);
            }
            vinCharImg.erase(vinCharImg.end() - zifuxu, vinCharImg.end());
            ffff = 1;
            zifuxu = 0;
            re_cut(matimg, nimg, threshold, sw, diji, 0, &zifuxu, vinCharImg);
          }
          if (zifuxu < 17 && ffff == 1) {
            for (int i = vinCharImg.size() - zifuxu; i < vinCharImg.size();
                 i++) {
              cvReleaseImage(&vinCharImg[i]);
            }
            vinCharImg.erase(vinCharImg.end() - zifuxu, vinCharImg.end());
          }
          if (zifuxu >= 17) {
            if (everycharNum.size() != 0) {
              if (zifuxu > everycharNum[0]) {

                for (int i = vinCharImg.size() - zifuxu; i < vinCharImg.size();
                     i++) {
                  cvReleaseImage(&vinCharImg[i]);
                }
                vinCharImg.erase(vinCharImg.end() - zifuxu, vinCharImg.end());

              } else {
                for (int i = 0; i < everycharNum[0]; i++) {
                  cvReleaseImage(&vinCharImg[i]);
                }
                vinCharImg.erase(vinCharImg.begin(),
                                 vinCharImg.begin() + everycharNum[0]);
                everycharNum.push_back(zifuxu);
              }
            } else {
              everycharNum.push_back(zifuxu);
            }
          }
          cvReleaseImage(&binaryimg);
          cvReleaseImage(&nimg);
        }
      }
      cvReleaseImage(&img_cc);
      cvResetImageROI(img_Clone1);
    }
    k++;
    contour = contour->h_next;
  }
  vector<int>(everycharNum).swap(everycharNum);
  cvReleaseMemStorage(&storage);
  ;
  // cvReleaseImage(&img);
  cvReleaseImage(&img_Clone1);
  return det;
}
//找盖章处参数cishu=1表示第二次查找
void FindMark(IplImage *src, IplImage *img, int cishu, IplImage *dst) {
  CvSeq *contour = NULL;
  CvMemStorage *storage = cvCreateMemStorage(0);
  int w = img->width;
  int h = img->height;
  IplImage *img1 = cvCreateImage(cvSize(w, h), 8, 1);
  cvCopy(img, img1);
  vector<CvRect> rectangle0; //保存适合的矩形区域
  vector<CvRect> rectangle;  //保存适合的矩形区域

  for (int j = 0; j < h; j++) {
    cvSet2D(img1, j, 0, cvScalar(0));
    cvSet2D(img1, j, w - 1, cvScalar(0));
  }
  //________
  for (int i = 0; i < w; i++) {
    cvSet2D(img1, 0, i, cvScalar(0));
    cvSet2D(img1, h - 1, i, cvScalar(0));
  }
  // cvShowImage("SD",img);
  cvFindContours(img1, storage, &contour, sizeof(CvContour), CV_RETR_LIST,
                 CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  cvReleaseImage(&img1);
  while (contour) {

    float tmparea = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
    CvRect rect = ((CvContour *)contour)->rect;
    contour = contour->h_next;
    float atio;
    atio = (float)rect.width / rect.height;
    long area = rect.width * rect.height;
    CvPoint p1, p2;
    p1.x = rect.x;
    p1.y = rect.y;
    p2.x = rect.x + rect.width;
    p2.y = rect.y + rect.height;
    if (rect.width > 30 || rect.height > 30) {
      rectangle0.push_back(rect);
    }
    //删除面积、高宽、长宽比、离图边距离近的区域rect.width>20||rect.height>180||rect.height<30||tmparea<10
    if (rect.y < 2 || (rect.x < 2 && rect.y < rect.height / 3)) {
      continue;
    }
    if (atio > 0.5 && atio < 2 && area > 5000 && rect.width < w / 2 &&
        rect.width > 50 && rect.height < h / 2 && rect.height > 80) {
      // cout<<"面积："<<area<<endl;
      rectangle.push_back(rect);
      // cvRectangle(src,p1,p2,CV_RGB(0,0,255),1,8,0);
      // cvDrawContours(min_img, contour, cvScalarAll(0),cvScalarAll(0), 0,
      // CV_FILLED,8);
    }
  }
  cvReleaseMemStorage(&storage);
  //找适合的矩形区域
  IplImage *r_img = NULL;
  CvPoint p3, p4; //记录切割区域的左上角和右下角坐标
  int flag = 0; //旋转标志：=1逆时针90度；=2顺时针90度；=3顺时针180度
  int flag1 = 0; //如果空区域则不旋转=1
                 //如果不存在区域则找整个证件区域
  // printf("第二次找区域\n");
  if (rectangle.size() < 1) {
    // printf("区域空\n");
    flag1 = 1;
    if (!(rectangle0.size() < 1 || cishu == 0)) {
      int up = h, down = 0, lift = w, ri = 0;
      for (int j = 0; j < rectangle0.size(); j++) {
        CvRect r0 = rectangle0[j];
        up = min(up, r0.y);
        lift = min(lift, r0.x);
        down = max(down, r0.y + r0.height);
        ri = max(ri, r0.x + r0.width);
      }
      cvSetImageROI(src,
                    cvRect(lift, up, abs(ri - lift + 1), abs(down - up + 1)));
      r_img =
          cvCreateImage(cvSize(abs(ri - lift + 1), abs(down - up + 1)), 8, 3);
      cvCopy(src, r_img);
    }
  } else //如果存在，则切割出:(rectangle.empty()==NULL)
  {
    // printf("存在区域\n");
    //目标区域,,如果所标区域只有一个
    // printf("%d\n",rectangle.size());
    CvRect r = rectangle[0];
    //所确定的目标超过一个
    // printf("标志区域11111\n");
    if (rectangle.size() > 1) {
      int xx = 0, yy = 0;
      for (int i = 1; i < rectangle.size(); i++) {
        CvRect r1 = rectangle[i];
        xx = max(r.x + r.width, r1.x + r1.width);
        yy = max(r.y + r.height, r1.y + r1.height);
        r.x = min(r1.x, r.x);
        r.y = min(r1.y, r.y);
        r.width = xx - r.x;
        r.height = yy - r.y;
      }
      //	printf("标志区域\n");
    }
    // printf("标志区域\n");
    if (r.x < w / 3) {
      // printf("标志区域小于1/3\n");
      //逆时针第一区域
      if (r.y < h / 3) {
        p3.x = 20;
        p3.y = r.y;
        p4.x = r.x + r.width + 20;
        p4.y = h - 20;
        if (r.width < w / 3) {
          flag = 1;
        }
        //旋转标志
        if (r.width > w / 2) {
          p3.x = 20;
          p3.y = r.y;
          p4.x = w - 20;
          p4.y = r.y + r.height + 30;
          if (r.y + r.height + 30 > h - 1) {
            p4.y = r.y + r.height;
          }
          //旋转标志
          if (r.height < h / 2) {
            flag = 3;
          }
          // cout<<"oooooooooooo"<<flag<<endl;
        }
        if (r.height >= h / 2) {
          p3.x = 20;
          p3.y = r.y;
          p4.x = w - 20;
          p4.y = r.y + r.height - 20;
          flag = 0; // if(r.y+r.height<h);
          // cout<<"dsfsdsd<w/3"<<endl;
        }

      } else {
        p3.x = r.x;
        p3.y = r.y - 30;
        if (r.height < 150 && r.y > h * 2 / 3) {
          p3.y = r.y - 100;
        } // if()
        p4.x = w - 20;
        p4.y = r.y + r.height + 10;
        if (p4.y > h - 1) {
          p4.y = h - 20;
        }
        if (p4.y < h * 2 / 3) {
          p4.y = h * 2 / 3 + 20;
        }
      }
    }
    if (r.x >= w / 3) {
      // printf("标志区域大于1/3\n");
      if (r.y < h / 2 && r.height < h / 2) {
        p3.x = 10;
        p3.y = r.y;
        p4.x = r.x + 10;
        p4.y = r.y + r.height + 30; //
        flag = 3;
        // cout<<"______________"<<flag<<endl;
      } else {
        p3.x = 10;
        p3.y = r.y - 10;
        p4.x = w - 10;
        p4.y = h - 20;
      }

      if (r.height > h / 3) {
        p3.x = 20;
        p3.y = r.y;
        p4.x = w - 20;
        p4.y = r.y + r.height - 10;
        // flag=2;
      }
      // flag=2情况
      if (r.y > h * 2 / 3 && (w - (r.x + r.width)) < 4) {
        flag = 2;
        p3.x = r.x - 20;
        p3.y = 10;
        p4.x = r.x + r.width;
        p4.y = r.y + r.height - 20;
      }
    }
    cvSetImageROI(src, cvRect(p3.x, p3.y, abs(p4.x - p3.x), abs(p4.y - p3.y)));
    r_img = cvCreateImage(cvSize(abs(p4.x - p3.x), abs(p4.y - p3.y)), 8, 3);
    cvCopy(src, r_img);
    float bii = (float)abs(p4.x - p3.x) / abs(p4.y - p3.y);
    if (bii < 0.34) {
      flag = 1;
    }
  }

  //旋转旋转标志：=1逆时针90度；=2顺时针90度；=3顺时针180度
  // clockwise 为true则顺时针旋转，否则为逆时针旋
  if (flag1 == 0 && flag == 1) {
    dst = rotateImage(r_img, 90, FALSE);
  }
  if (flag1 == 0 && flag == 2) {
    dst = rotateImage(r_img, 90, TRUE);
  }
  if (flag1 == 0 && flag == 3) {
    IplImage *dst1 = NULL;
    dst1 = rotateImage(r_img, 90, TRUE);
    dst = rotateImage(dst1, 90, TRUE);
    cvReleaseImage(&dst1);
  }
  cvReleaseImage(&r_img);
  vector<CvRect>(rectangle).swap(rectangle);
  vector<CvRect>(rectangle0).swap(rectangle0);
}
//第一次识别二值化函数定位用
void process(IplImage *src, IplImage *binimg) {
  IplConvKernel *KR1 = cvCreateStructuringElementEx(2, 1, 0, 0, CV_SHAPE_RECT);
  IplConvKernel *KR2 = cvCreateStructuringElementEx(1, 2, 0, 0, CV_SHAPE_RECT);
  IplConvKernel *KR3 = cvCreateStructuringElementEx(3, 1, 0, 0, CV_SHAPE_RECT);

  IplImage *img1 =
      cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1); //创建灰度图像
  // IplImage* binimg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U,
  // 1);//创建目标图像
  cvCvtColor(src, img1, CV_BGR2GRAY);
  //_______________________
  cvSmooth(img1, img1, CV_GAUSSIAN, 3, 3);
  cvSobel(img1, img1, 1, 0, 3);
  int bin = otsu(img1);
  // cout<<bin<<endl;
  cvThreshold(img1, binimg, bin, 255, CV_THRESH_BINARY); // cvNot(binimg,binimg);
  // cvShowImage("22222域原图",binimg);

  // IplImage* binimg1 = cvCreateImage(cvGetSize(binimg), IPL_DEPTH_8U, 1);
  DeleteMinarea0(binimg);
  cvDilate(binimg, binimg, KR3, 1);
  cvDilate(binimg, binimg, KR1, 12);
  cvErode(binimg, binimg, KR1, 1);
  cvErode(binimg, binimg, KR2, 3);
  // cvErode(binimg,binimg,KR2,6);
  cvDilate(binimg, binimg, KR2, 3);
  cvDilate(binimg, binimg, KR1, 1);
  // cvShowImage("xiangtai原图",binimg);
  // ExtrVin(src,binimg);
  // cvWaitKey(0);
  // cvReleaseImage(&src);
  cvReleaseImage(&img1);
  cvReleaseStructuringElement(&KR1);
  cvReleaseStructuringElement(&KR2);
  cvReleaseStructuringElement(&KR3);

  //	return binimg;
}
//第二次识别二值化函数__找盖章处
void process1(IplImage *src, IplImage *binimg) {
  // char file_dst[100];
  IplConvKernel *KR1 = cvCreateStructuringElementEx(1, 2, 0, 0, CV_SHAPE_RECT);
  IplConvKernel *KR11 = cvCreateStructuringElementEx(2, 1, 0, 0, CV_SHAPE_RECT);
  // IplImage *src1;
  IplImage *img1 =
      cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1); //创建目标图像
  IplImage *img2 =
      cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1); //创建目标图像
  // IplImage* binimg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U,
  // 1);//创建目标图像
  cvCvtColor(src, img1, CV_BGR2GRAY);
  cvSmooth(img1, img1, CV_GAUSSIAN, 5, 5);
  // cvEqualizeHist(img1,img1);
  cvSobel(img1, img2, 1, 0, 3);   //垂直
  cvSobel(img1, binimg, 0, 1, 3); //水平
  int im = otsu(img2) - 0;
  int bin = otsu(binimg) - 0;
  cvThreshold(
      binimg, binimg, bin, 255,
      CV_THRESH_BINARY); // cvNot(binimg,binimg);|CV_THRESH_OTSU|CV_THRESH_OTSU
  // sprintf(file_dst,"E:\\miss\\%s.png",imgName);
  // IplImage* binimg1 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
  DeleteMinarea1(binimg);
  // cvShowImage("11111域原图",binimg);
  cvThreshold(img2, img2, im, 255, CV_THRESH_BINARY);
  // cvShowImage("22域原图",img2);
  // IplImage* img3 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
  DeleteMinarea2(img2);
  // cvShowImage("22222域原图",img2);
  plusimg(img2, binimg);
  //
  // cvDilate(binimg,binimg,KR3,1);
  cvDilate(binimg, binimg, KR1, 2);
  cvDilate(binimg, binimg, KR11, 2); // cvShowImage("边缘",binimg);
  // sprintf(file_dst,"E:\\miss\\%s.tif",imgName);
  // cvSaveImage(file_dst,binimg);
  cvReleaseImage(&img1);
  cvReleaseImage(&img2);

  cvReleaseStructuringElement(&KR1);
  cvReleaseStructuringElement(&KR11);
}
//第二次识别用二值化函数定位用
IplImage *process2(IplImage *src) {
  IplConvKernel *KR1 = cvCreateStructuringElementEx(2, 1, 0, 0, CV_SHAPE_RECT);
  // IplConvKernel* KR2=cvCreateStructuringElementEx(1,2,0,0,CV_SHAPE_RECT);
  IplImage *img1 =
      cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1); //创建灰度图像
  IplImage *binimg =
      cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1); //创建目标图像
  cvCvtColor(src, img1, CV_BGR2GRAY);
  cvSmooth(img1, img1, CV_GAUSSIAN, 5, 5);
  cvSobel(img1, img1, 0, 1, 3);
  int bin = otsu(img1);

  cvThreshold(img1, binimg, bin, 255, CV_THRESH_BINARY); // cvNot(binimg,binimg);
  cvDilate(binimg, binimg, KR1, 5);
  cvReleaseStructuringElement(&KR1);
  cvReleaseImage(&img1);
  return binimg;
}
//判断是否旋转
//判断是否旋转返回0 旋转；1 逆时针；2顺时针；
int ifrotate(IplImage *src) {
  int flag = 0;
  //转为灰度图像huid
  IplImage *huid = cvCreateImage(cvGetSize(src), 8, 1);
  cvCvtColor(src, huid, CV_BGR2GRAY);
  //求垂直边缘
  IplImage *sobel = cvCreateImage(cvGetSize(src), 8, 1);
  cvSobel(huid, sobel, 1, 0, 3);
  cvThreshold(sobel, sobel, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);
  // cvShowImage("垂直边缘",sobel);
  //形态学处理
  // IplConvKernel* KR3=cvCreateStructuringElementEx(1,5,0,0,CV_SHAPE_RECT);
  IplConvKernel *KR2 = cvCreateStructuringElementEx(1, 2, 0, 0, CV_SHAPE_RECT);
  // cvErode(sobel,sobel,KR2,1);

  cvDilate(sobel, sobel, KR2, 6);
  // cvShowImage("垂直边缘xingti0",sobel);
  // cvShowImage("垂直边缘xingti1",sobel);
  // cvErode(sobel,sobel,KR2,1);

  for (int i = 0; i < sobel->width; i++) {
    cvSet2D(sobel, 0, i, cvScalar(0));
    cvSet2D(sobel, sobel->height - 1, i, cvScalar(0));
  }
  int maxh = 0, xx = 0, maxy = 0, yx = 0, minx = src->width, maxx = 0;
  CvSeq *contour = NULL;
  CvMemStorage *storage = cvCreateMemStorage(0);
  cvFindContours(sobel, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL,
                 CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  int k = 0; //垂直线的数目
  while (contour) {
    CvRect rect = ((CvContour *)contour)->rect;
    CvPoint p1, p2;
    p1.x = rect.x;
    p1.y = rect.y;
    p2.x = rect.x + rect.width;
    p2.y = rect.y + rect.height;
    //&&rect.y+rect.height>src->height/2
    if (rect.height > 100 && abs(rect.x - src->width / 2) < src->width / 8 &&
        rect.width < 30) {
      k++;
      // cvRectangle(src,p1,p2,CV_RGB(255,0,0),1,8,0);
      flag = 1; // 1 逆时针；2顺时针；
      maxx = max(maxx, rect.x); //最大x值
      maxy = max(maxy, rect.y + rect.height);
      if (maxy == (rect.y + rect.height)) {
        yx = rect.x;
      }
      minx = min(minx, rect.x);
      maxh = max(maxh, rect.height);
      if (maxh == rect.height) {
        xx = rect.x;
      }
      // if(minx<src->width/2-40){break;}maxh>3*src->height/4&&xx<src->width/2-40||
      //逆时针旋转图像
      if (src->height - maxy < src->height / 5 && yx > 2 * src->width / 3) {
        flag = 2;
      }
      if (k < 3 || maxy < 3 * src->height / 5) {
        flag = 0;
      }
    }
    contour = contour->h_next;
  }
  cvReleaseMemStorage(&storage);
  cvReleaseImage(&huid);
  cvReleaseImage(&sobel);
  cvReleaseStructuringElement(&KR2);
  return flag;
}
//第一次识别尺度归一化
IplImage *resize(IplImage *src) {
  int w = src->width;
  int h = src->height;
  float wh = (float)w / h;

  IplImage *dst1 = NULL, *dst = NULL;
  if (wh >= 1) {
    int hh = (int)720. / wh;
    dst1 = cvCreateImage(cvSize(720, hh), 8, 3);
  } else {
    int ww = (int)720. * wh;
    dst1 = cvCreateImage(cvSize(ww, 720), 8, 3);
  }
  cvResize(src, dst1);
  //判断是否旋转返回0 旋转；2顺时针；1 逆时针；
  // cout<<"dddddd"<<endl;
  int f = ifrotate(dst1);
  // clockwise 为true则顺时针旋转，否则为逆时针旋转
  if (f != 0) {
    IplImage *dst2 = NULL;
    // cout<<"第"<<endl;
    dst2 = rotateImage(dst1, 90, FALSE);
    int f1 = ifrotate(dst2);
    // cout<<"sss"<<f1<<endl;
    if (f1 != 0) {
      IplImage *dst3 = NULL;
      dst3 = rotateImage(dst2, 90, TRUE);
      CvRect rect = cvRect(dst3->width / 5 + 30, dst3->height / 5 + 20,
                           dst3->width - dst3->width / 5 - 60,
                           dst3->height - dst3->height / 5 - 80);
      cvSetImageROI(dst3, rect);
      float atio = (float)dst3->height / dst3->width;
      dst = cvCreateImage(cvSize(480, 480 * atio), 8, 3);
      cvResize(dst3, dst);
      cvReleaseImage(&dst3);
    } else {
      CvRect rect = cvRect(dst2->width / 5 + 30, dst2->height / 5 + 20,
                           dst2->width - dst2->width / 5 - 60,
                           dst2->height - dst2->height / 5 - 80);
      cvSetImageROI(dst2, rect);
      float atio = (float)dst2->height / dst2->width;
      dst = cvCreateImage(cvSize(480, 480 * atio), 8, 3);
      cvResize(dst2, dst);
    }
    cvReleaseImage(&dst2);
  } else {
    CvRect rect = cvRect(dst1->width / 5 + 30, dst1->height / 5 + 20,
                         dst1->width - dst1->width / 5 - 60,
                         dst1->height - dst1->height / 5 - 80);
    cvSetImageROI(dst1, rect);
    float atio = (float)dst1->height / dst1->width;
    dst = cvCreateImage(cvSize(480, 480 * atio), 8, 3);
    cvResize(dst1, dst);
  }
  cvReleaseImage(&dst1);
  return (dst);
}
//第二次识别用的尺度变换
IplImage *resize1(IplImage *src) {
  int w = src->width;
  int h = src->height;
  float wh = (float)w / h;
  IplImage *dst1 = NULL;
  if (wh >= 1) {
    dst1 = cvCreateImage(cvSize(720, 720. / wh), 8, 3);
  }
  if (wh < 1) {
    dst1 = cvCreateImage(cvSize(720 * wh, 720), 8, 3);
  }
  cvResize(src, dst1);
  //判断是否旋转返回0 旋转；2顺时针；1 逆时针；
  int f = ifrotate(dst1);
  // clockwise 为true则顺时针旋转，否则为逆时针旋转
  if (f != 0) {
    IplImage *dst2 = NULL;
    dst2 = rotateImage(dst1, 90, FALSE);
    int f1 = ifrotate(dst2);
    if (f1 != 0) {
      IplImage *dst3 = NULL;
      dst3 = rotateImage(dst2, 90, TRUE);
      cvReleaseImage(&dst1);
      cvReleaseImage(&dst2);
      return (dst3);
    }
    cvReleaseImage(&dst1);
    return (dst2);
  } else {

    return (dst1);
  }
}
//分割后的字符进行识别
void FeatureExtraction(IplImage *src, CvMat *data_mat, int ind) //
{
  IplImage *trainImg = cvCreateImage(
      cvSize(20, 60), 8,
      3);                  //需要分析的图片，这里默认设定图片是64*64大小，所以上面定义了1764，如果要更改图片大小，可以先用debug查看一下descriptors是多少，然后设定好再运行
  cvResize(src, trainImg); //读取图片
  HOGDescriptor *hog =
      new HOGDescriptor(cvSize(20, 60), cvSize(2, 4), cvSize(2, 4),
                        cvSize(2, 4), 9); //具体意思见参考文章1,2
  vector<float> descriptors;              //结果数组e

  hog->compute(trainImg, descriptors, Size(1, 1),
               Size(0, 0)); //调用计算函数开始计算
  // cout<<"nidaye"<<endl ;
  int n = 0;
  for (vector<float>::iterator iter = descriptors.begin();
       iter != descriptors.end(); iter++) {
    cvmSet(data_mat, ind, n, *iter); //把HOG存储下来
    n++;
  }
  vector<float>(descriptors).swap(descriptors);
  cvReleaseImage(&trainImg);
  // cout<<SVMtrainMat->rows<<endl;
}
//识别
bool recvin(cv::Mat src_mat, char *pre, CvSVM *svmvin, CvSVM *svm33, CvSVM *svm35, CvSVM *svmLast5, const char * imgpath) {
  int k = 0;
  string charList = "0123456789ABCDEFGHJKLMNPRSTUVWXYZIQ"; //字符
  IplImage *img = NULL;
  IplImage *src = NULL;
  IplImage ipl_img(src_mat);
  src = &ipl_img;
  img = resize(src);
  // cvShowImage("ssqqqs",img);
  // cvWaitKey(0);
  //得到二值化图片
  IplImage *binimg0 = cvCreateImage(cvGetSize(img), 8, 0);
  process(img, binimg0);

  //提取Vin码并进行字符分割,分割后的字符存在vinCharImg中
  //清空上次保留的数据
  vector<IplImage *> vinCharImg;
  vinCharImg.clear();
  bool det = false;
  cvSaveImage(imgpath, img);
  det = ExtrVin(img, binimg0, svmvin, vinCharImg);
  cvReleaseImage(&binimg0);
  cvReleaseImage(&binimg0);
  //第一次定位结束
  if (vinCharImg.size() == 0) {
    det = FALSE;
  }
  IplImage *r_img = NULL;

  if (!det) //第一次定位失败，则开始第二次定位
  {
    printf("第二次定位\n");

    IplImage *img1 = NULL;
    img1 = resize1(src);
    IplImage *binimg1 = cvCreateImage(cvGetSize(img1), 8, 0);
    process1(img1, binimg1);
    //	printf("第二次定位二值化结束\n");
    int cishu = 0;
    int ifregion = 1; //标记是否查找vin码区域程
    FindMark(img1, binimg1, cishu, r_img);
    // printf("第二次第一次定位\n");
    if (r_img == NULL) {
      cishu = 1;
      IplConvKernel *KR1 =
          cvCreateStructuringElementEx(2, 2, 0, 0, CV_SHAPE_RECT);

      cvDilate(binimg1, binimg1, KR1, 8);
      FindMark(img1, binimg1, cishu, r_img);
      cvReleaseStructuringElement(&KR1);
    }
    cvReleaseImage(&binimg1);
    cvReleaseImage(&img1);
    // cvShowImage(imgName,r_img);
    //	cvWaitKey(0);
    if (r_img != NULL) {
      IplImage *binimg2 = cvCreateImage(cvGetSize(r_img), 8, 0);
      binimg2 = process2(r_img);

      vinCharImg.clear();
      cvSaveImage(imgpath, r_img);
      det = ExtrVin1(r_img, binimg2, svmvin, vinCharImg);
      cvReleaseImage(&binimg2);
    }
    //______________________________第二次定位结束______________________________________________
  }
  // cout<<"开始识别"<<endl;
  int flag = 0;
  bzero(pre, sizeof(pre));
  if (vinCharImg.size() > 0) {
    CvMat *data_mat;
    data_mat = cvCreateMat(1, 1350, CV_32FC1);
    cvSetZero(data_mat); //初始化为0
    int res;

    int numFu = 0;
    // cout<<vinCharImg.size()<<endl;
    for (int charInd = 0; charInd != (int)vinCharImg.size(); charInd++) {
      IplImage *charImg = vinCharImg[charInd];
      if (charImg == NULL) {
        continue;
      }

      flag = 1;
      FeatureExtraction(charImg, data_mat, 0);
      // printf("到这里了sssssssssssssssssss\n");
      if (vinCharImg.size() == 17) {
        res =
            charInd < 12 ? svm33->predict(data_mat) : svmLast5->predict(data_mat);

      } else {
        res =
            charInd < 12 ? svm35->predict(data_mat) : svmLast5->predict(data_mat);
      }
      if (res >= 33 && numFu < vinCharImg.size() - 17) {
        numFu++;
        cvReleaseImage(&charImg);
        continue;
      }

      //cerr << "000[" << res << "]k={" << k << "}" << endl;
      // cout<<charList[res];
      pre[k] = (char)charList[res];
      k++;

      cvReleaseImage(&charImg);
    }

    cvReleaseMat(&data_mat);
  }
  //未识别成
  //                     printf("到这里了xxxxxxxxx：%s\n");

  if (flag == 0) {
    // cout<<"识别失败"<<endl;
    k = 4;
    strcpy(pre, "Fail");
    return false;
  }
  cvReleaseImage(&r_img);
  return true;
  // cvReleaseImage(&src);
}

/*
//主函数
int main(int argc, char *argv[]) {
  //识别程序
  cout << "开始下载训练数据" << endl;
  CvSVM svmvin;
  CvSVM svm33;
  CvSVM svm35;
  CvSVM svmLast5;
  svmvin.load("SVM_D.xml");
  svm35.load("SVM_DATA_all0-Q_1000_LINEAR.xml");
  svm33.load("SVM_DATA_all0-Z_1000_LINEAR.xml");
  svmLast5.load("SVM_DATA_all0-9_1000_LINEAR.xml"); //得到字符识别svm
  cv::Mat image = imread("licensefimg.jpg", IMREAD_COLOR);
  // imshow("sss",image);
  // cvWaitKey(0);
  char pre[30] = ""; // save recgnizition result
  recvin(image, pre, &svmvin, &svm33, &svm35, &svmLast5);
  cout << pre << endl;
  return (0);
}
*/

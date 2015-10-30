#include <stdlib.h>
#include <sys/types.h>
#include "highgui.h"
#include <stdio.h>
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

using namespace cv;
using namespace std;

//最大类间方差阈值（大津）法
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
//使用大津法：水平方向分段二值化（k：分几段）__防止图片部分光照带来的影响；src为原图，binimg为分段后的图片
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
//二值图像相加__(如用于sobel的垂直和水平边缘相加)
void plusimg(IplImage *img, IplImage *dst) {
  for (int i = 0; i < img->height; i++) {
    for (int j = 0; j < img->width; j++) {
      if (CV_IMAGE_ELEM(img, uchar, i, j) > 100) {
        CV_IMAGE_ELEM(dst, uchar, i, j) = 255;
      }
    }
  }
}
//判断是否旋转（根据垂直边缘：如果图像中间位置垂直边缘存在竖直直线，则旋转）返回
// 0不旋转；1逆时针；2顺时针；
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
  // cvShowImage("垂直边缘xingti0",sobel);
  cvDilate(sobel, sobel, KR2, 6);

  //
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
  // cvShowImage("垂直边缘xingti1",sobel);
  int k = 0; //垂直线的数目
  while (contour) {
    CvRect rect = ((CvContour *)contour)->rect;
    // CvPoint p1,p2; p1.x=rect.x; p1.y=rect.y; p2.x=rect.x + rect.width;
    // p2.y=rect.y + rect.height;
    //&&rect.y+rect.height>src->height/2
    if (rect.height > 100 && abs(rect.x - src->width / 2) < src->width / 8 &&
        rect.width < 30) {
      k++;
      // cvRectangle(src,p1,p2,CV_RGB(255,0,0),1,8,0);
      flag = 1;                 // 1 逆时针；2顺时针；
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
    }
    contour = contour->h_next;
  }
  if (k < 3 || maxy < 3 * src->height / 5) {
    flag = 0;
  }
  cvReleaseMemStorage(&storage);
  cvReleaseImage(&huid);
  cvReleaseImage(&sobel);
  cvReleaseStructuringElement(&KR2);
  return flag;
}
//对图片旋转，clockwise 为true则顺时针旋转，否则为逆时针旋转
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
//删除非感兴趣区域Region_of_non_interest__根据连通区域宽、高、宽高比、面积
void DeleteRONI(IplImage *img, int maxw, int minw, int maxh, int minh,
                float maxwhatio, float minwhatio, float maxarea,
                float minarea) {
  CvSeq *contour = NULL;
  CvMemStorage *storage = cvCreateMemStorage(0);
  IplImage *min_img = cvCreateImage(cvGetSize(img), 8, 1);
  cvCopy(img, min_img);
  //消除边缘带来的影响
  for (int j = 0; j < img->height; j++) {
    cvSet2D(min_img, j, 0, cvScalar(0));
    cvSet2D(min_img, j, img->width - 1, cvScalar(0));
  }
  for (int i = 0; i < img->width; i++) {
    cvSet2D(min_img, 0, i, cvScalar(0));
    cvSet2D(min_img, img->height - 1, i, cvScalar(0));
  }
  //找轮廓___
  cvFindContours(min_img, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP,
                 CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  cvReleaseImage(&min_img);
  while (contour) {
    float tmparea = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
    CvRect rect = ((CvContour *)contour)->rect;
    float atio = (float)rect.width / rect.height;
    //删除不满足阈值区域
    if (rect.width > maxw || rect.width < minw || rect.height > maxh ||
        rect.height < minh || tmparea < minarea || tmparea > maxarea ||
        atio < minwhatio || atio > maxwhatio) {
      cvDrawContours(img, contour, cvScalarAll(0), cvScalarAll(0), 0, CV_FILLED,
                     8);
    }
    contour = contour->h_next;
  }
  cvReleaseMemStorage(&storage);
}
//尺度归一化,and图片剪切——找证件区域
IplImage *jianqie0(IplImage *src) {
  //尺度归一化————————————————————————————————
  int w = src->width;
  int h = src->height;
  //_____________________________________________________________________________________
  IplImage *hui = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
  cvCvtColor(src, hui, CV_BGR2GRAY); // 转为灰度图像
  //创建目标二值图像
  IplImage *binimg =
      cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1); //垂直方向裁剪
  cvSobel(hui, hui, 1, 0, 3);                         //垂直边缘
  cvThreshold(hui, binimg, 0, 255,
              CV_THRESH_BINARY | CV_THRESH_OTSU); //进行二值化
  //处理边缘
  for (int j = 0; j < h; j++) {
    cvSet2D(binimg, j, 0, cvScalar(0));
    cvSet2D(binimg, j, w - 1, cvScalar(0));
  }
  for (int i = 0; i < w; i++) {
    cvSet2D(binimg, 0, i, cvScalar(0));
    cvSet2D(binimg, h - 1, i, cvScalar(0));
  }
  // 1)_____求切割区域的上下坐标点
  CvSeq *contour = NULL;
  CvMemStorage *storage = cvCreateMemStorage(0); //垂直
  int upymin = binimg->height; // upymin记录最上边缘点y坐标;
  int downy = 0; // downy记录剪切的最低边缘点,（？downh为对应的高度）;
  int x1 = binimg->width; //记录最小x
  int x2 = 0;             //记录最大x;
                          //扫描方式：从下往上，从右往左
  // cvShowImage("ppp",binimg);
  cvFindContours(binimg, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL,
                 CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  cvReleaseImage(&binimg);
  int ff0 = 0;
  while (contour) {
    int ff0 = 1;
    CvRect rect1 = ((CvContour *)contour)->rect;
    float tmparea = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
    contour = contour->h_next;
    int xw = rect1.x + rect1.width;
    if (tmparea > 10) {
      x2 = max(x2, xw);
      x1 = min(x1, rect1.x);
    }
    if (rect1.height > src->height - 10) {
      continue;
    }
    if (rect1.height < 40) {
      continue;
    }
    int yh = rect1.y + rect1.height; //为了使式子简单;
    downy = max(yh, downy);
    upymin = min(upymin, rect1.y);
  }
  cvReleaseMemStorage(&storage);
  cvReleaseImage(&hui);
  //求出切割上下坐标点
  if (ff0 == 0) {
    x2 = w;
    x1 = 0;
  }
  int ht1 = downy - upymin;
  int wt1 = x2 - x1;
  if (ht1 < 60) {
    upymin = 5;
    ht1 = h - 10;
  }
  if (wt1 < 60) {
    x1 = 5;
    wt1 = w - 10;
  }
  cvSetImageROI(src, cvRect(x1, upymin, wt1, ht1));
  IplImage *img = cvCreateImage(cvSize(wt1, ht1), 8, 3);
  cvCopy(src, img);

  // cvShowImage("ssss",img);

  //判断是否旋转_________________________________________________________
  int f = ifrotate(img);
  if (f != 0) {
    img = rotateImage(img, 90, TRUE);
  }
  // 2)____________________________求水平左右切割点
  IplImage *hui1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
  cvCvtColor(img, hui1, CV_BGR2GRAY);
  //创建目标二值图像
  IplImage *binimg1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
  cvSobel(hui1, hui1, 0, 1, 3);
  cvThreshold(hui1, binimg1, 0, 255,
              CV_THRESH_BINARY | CV_THRESH_OTSU); //水平方向裁剪
  IplConvKernel *KR1 = cvCreateStructuringElementEx(2, 1, 0, 0, CV_SHAPE_RECT);
  cvDilate(binimg1, binimg1, KR1, 2);
  cvReleaseStructuringElement(&KR1);
  // cvShowImage("wwww",binimg1);
  for (int j = 0; j < binimg1->height; j++) {
    cvSet2D(binimg1, j, 0, cvScalar(0));
    cvSet2D(binimg1, j, binimg1->width - 1, cvScalar(0));
  }
  for (int i = 0; i < binimg1->width; i++) {
    cvSet2D(binimg1, 0, i, cvScalar(0));
    cvSet2D(binimg1, binimg1->height - 1, i, cvScalar(0));
  }
  CvSeq *contour1 = NULL;
  CvMemStorage *storage1 = cvCreateMemStorage(0); //垂直
  int minx = binimg1->width; // minx记录最右边缘点y坐标;
  int maxx = 0; //记录剪切的最左边边缘点,（？为对应的高度）;
  int y1 = binimg1->height;
  int y2 = 0;
  //扫描方式：从下往上，从右往左
  cvFindContours(binimg1, storage1, &contour1, sizeof(CvContour),
                 CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
  cvReleaseImage(&binimg1);
  int ff1 = 0;
  while (contour1) {
    CvRect rect2 = ((CvContour *)contour1)->rect;
    contour1 = contour1->h_next;
    ff1 = 1;
    if (rect2.width > img->width - 10) {
      continue;
    }
    if (rect2.width < 40) {
      continue;
    }
    int xw = rect2.x + rect2.width; //为了使式子简单;maxH=rect1.height;//
    // maxH=downy-upymin;
    maxx = max(maxx, xw);
    minx = min(minx, rect2.x);
    y1 = min(y1, rect2.y);
    y2 = max(y2, (rect2.y + rect2.height));
  }
  cvReleaseMemStorage(&storage1);
  cvReleaseImage(&hui1);
  //求出切割左右坐标点
  if (ff1 == 0) {
    y2 = img->height;
    y1 = 0;
  }
  int wt2 = maxx - minx;
  int ht2 = y2 - y1;
  if (ht2 < 60) {
    y1 = 5;
    ht2 = img->height - 10;
  }
  if (wt2 < 60) {
    minx = 5;
    wt2 = img->width - 10;
  }
  cvSetImageROI(img, cvRect(minx, y1, wt2, ht2));
  IplImage *dst = cvCreateImage(cvSize(wt2, ht2), 8, 3);
  cvCopy(img, dst);
  cvReleaseImage(&img);
  return (dst);
}
IplImage *charprocess0(IplImage *src) {
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
  int downy = 0,
      downh = 0; // downy记录剪切的最低边缘点,（？downh为对应的高度）;
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
    if (rect1.height < 11) {
      continue;
    }
    int yh = rect1.y + rect1.height; //为了使式子简单;maxH=rect1.height;//
    // maxH=downy-upymin;
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
int charprocess(IplImage *src, int diji, int &zifuxu,
                vector<IplImage *> &vinCharImg) {
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
  int downy = 0,
      downh = 0; // downy记录剪切的最低边缘点,（？downh为对应的高度）;
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
  zifuxu++;
  // cv::Mat matimg;
  // matimg=cv::Mat(src);
  // cout<<"sizeqian:"<<vinMatImg.size()<<endl;
  // vinMatImg.push_back(matimg);
  // cout<<"sizehou:"<<vinMatImg.size()<<endl;

  vinCharImg.push_back(src); //保存剪切后的字符
  //加识别程序___________________
  return (0);
}
//倾斜矫正：img原图,src需要矫正图片,rect记录src在img中的位置
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
  typedef unsigned char BYTE;
  BYTE pixel;
  double x1 = 0.0, y1 = 0.0;
  for (i = 0; i < h; i++) {
    for (j = 0; j < w; j++) {
      pixel = (BYTE)sobel2->imageData[i * nLinebyte + j];
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
      pixel = (BYTE)sobel2->imageData[i * nLinebyte + j];
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
//字符分割__计算vin码二值图片垂直投影。
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
//字符分割__比较每横坐标点垂直投影与阈值threshold，记录横坐标上的分割坐标
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
//在re_cut之后不满足分割宽度再次进行分割
void re_cutagain(cv::Mat img, IplImage *pI_1, int threshold, int sw, int diji,
                 int flag, int &zifuxu, vector<IplImage *> &vinCharImg) {
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
          zifuxu++;
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
          zifuxu++;
          vinCharImg.push_back(qie2);
        }
      }
      continue;
    }
    //保存分割出来的图片
    if (flag == 1) {
      charprocess(pI_3, diji, zifuxu, vinCharImg);
    } else {
      zifuxu++;
      vinCharImg.push_back(pI_3);
    }
  }
}
//字符分割，flag==1表示首次分割，其他表示首次分割字符数<17后的再次分割
void re_cut(cv::Mat img, IplImage *pI_1, int threshold, int sw, int diji,
            int flag, int &zifuxu, vector<IplImage *> &vinCharImg) {
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
      pic = charprocess0(pI_3);

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
      zifuxu++;

      vinCharImg.push_back(pI_3); //保存分
    }
  }
  // cvReleaseImage(&pI_3);
}
//找日期图片: rect标记Vin码的位置;flag_定位vin码是否发生了旋转
void FindDatePic(IplImage *img, CvRect rect, int flag,
                 const char path_dateimg[]) {
  //保存日期_________________________________________________________________________________________//////////
  IplImage *dateimg = NULL; //日期切割图片
  CvRect rect_date;         //标定日期的位置
  // cout<<"1==="<<rect.height<<rect.y<<endl;
  if (flag == 0) // ifrotate_flag=1代表vin位于左上角，180度旋转
  {
    if ((img->height - rect.y) < 6 * rect.height) {
      rect_date.y = min(2 * rect.height + rect.y, max(img->height - 30, 0));
      rect_date.x = max(rect.x - rect.width / 3, 0);
      rect_date.height = img->height - rect_date.y;
      rect_date.width = rect.width / 3 + rect.width / 2;
    } else {
      rect_date.y = rect.y + 3 * rect.height;
      rect_date.height = 3 * rect.height;
      rect_date.x = max(rect.x - rect.width / 3, 0);
      rect_date.width = rect.width / 3 + rect.width / 2;
    }
  } else {
    if (rect.y < 6 * rect.height) {
      rect_date.y = 0;
      rect_date.height = 3 * rect.height;
      if (rect_date.y + rect_date.height > img->height) {
        rect_date.height = img->height - rect_date.y;
      }
      rect_date.x = rect.x + rect.width / 2;
      rect_date.width = rect.width * 5 / 6;
      if (rect_date.x + rect_date.width > img->width) {
        rect_date.width = img->width - rect_date.x;
      }
    } else {
      rect_date.y = max(rect.y - 6 * rect.height, 0);
      rect_date.height = rect.height * 4;
      if (rect_date.y + rect_date.height > img->height) {
        rect_date.height = rect.y - rect_date.y;
      }
      rect_date.x = rect.x + rect.width / 2;
      rect_date.width = rect.width * 5 / 6;
      if (rect_date.x + rect_date.width > img->width) {
        rect_date.width = img->width - rect_date.x;
      }
    }
  }
  // cout<<"2==="<<rect_date.height<<endl;
  //保存日期图片
  dateimg = cvCreateImage(cvSize(rect_date.width, rect_date.height), 8, 3);
  cvSetImageROI(img, rect_date); //;cvWaitKey(0);
  cvCopy(img, dateimg);
  if (flag != 0) {
    dateimg = rotateImage(dateimg, 90, TRUE);
    dateimg = rotateImage(dateimg, 90, TRUE);
  }
  // DateOrFail_image=cv::Mat(dateimg,true);
  // save the image of date
  cvSaveImage(path_dateimg, dateimg);

  //
  cvReleaseImage(&dateimg);
  cvResetImageROI(img);
}
//第一次识别提取src为彩色图像；img为得到连通区域后的二值图像
bool ExtrVin(IplImage *src, IplImage *img, CvSVM *svmvin,
             vector<IplImage *> &vinCharImg, const char path_dateimg[]) {
  CvSeq *contour = NULL;
  CvMemStorage *storage = cvCreateMemStorage(0);
  IplImage *img_Clone = cvCreateImage(cvGetSize(img), 8, 1);
  IplImage *img_Clone1 = cvCreateImage(cvGetSize(src), 8, 3);

  bool det = false;

  char file_dst[200];      //保存路径
  cvCopy(src, img_Clone1); //归一化后的彩色图像
  cvCopy(img, img_Clone);  //得到的连通区域图片
  int k = 0;               //保存图片用
  det = FALSE;             //标记是否检测出正样本
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
    CvRect rect1, rect3; //获取感兴趣的矩形区
    rect3 = rect;
    rect1.x = max((rect.x - 3), 0);
    rect1.y = max((rect.y - 3), 0);
    rect1.width = (rect.x + rect.width + 6) < img->width ? (rect.width + 6)
                                                         : (rect.width);
    rect1.height = (rect.y + rect.height + 6) < img->height ? (rect.height + 6)
                                                            : (rect.height);
    // tmparea>100&&
    if (tmparea < 20000 && rect.width > 80 && rect.height < 120) {
      cvSetImageROI(img_Clone1, rect1); //获取感兴趣区域
      IplImage *img_c = cvCreateImage(cvSize(rect1.width, rect1.height), 8, 3);
      IplImage *img_cc = NULL;
      IplImage *xuanzhuan = NULL;
      cvCopy(img_Clone1, img_c); //重置感兴趣区域
      cvResetImageROI(img_Clone1);
      //倾斜矫正
      xuanzhuan = fcjiaodu(src, img_c, rect);
      cvReleaseImage(&img_c); //释放
      // cvShowImage("ss",xuanzhuan);
      // cvWaitKey(0);
      img_cc = charprocess0(xuanzhuan); //旋转
      cvReleaseImage(&xuanzhuan);       //释放

      if (img_cc !=
          NULL) ////开始分类检测____________________________________________________________________________
      {
        IplImage *trainImg = cvCreateImage(cvSize(232, 28), 8, 3);
        cvZero(trainImg);
        //判断是否旋转________________________________
        int ifrotate_flag = 0;
        if (rect.x < img->width / 4) //如果定位的位置在左上角则旋转图片180°
        {
          ifrotate_flag = 1;
          img_cc = rotateImage(img_cc, 90, TRUE);
          img_cc = rotateImage(img_cc, 90, TRUE);
        }
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

          // if(det==TRUE){break;}
          det = TRUE; //标志定位到了Vin码
          //切割日期
          if (diji == 0) {
            FindDatePic(src, rect, ifrotate_flag, path_dateimg);
          }

          //字符分割
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
          int zifuxu = 0; //对字符剪切后的字符数

          re_cut(matimg, nimg, threshold, sw, ++diji, 1, zifuxu, vinCharImg);
          // cout<<"sizeiplimag:"<<vinCharImg.size()<<endl;
          //如果分割出的字符数目少于17
          // cout<<zifuxu<<endl;

          int ffff = 0;

          if (zifuxu < 17) {
            for (int i = vinCharImg.size() - zifuxu; i < vinCharImg.size();
                 i++) {
              cvReleaseImage(&vinCharImg[i]);
            }
            vinCharImg.erase(vinCharImg.end() - zifuxu, vinCharImg.end());
            ffff = 1;
            zifuxu = 0;
            re_cut(matimg, nimg, threshold, sw, diji, 0, zifuxu, vinCharImg);
            // cout<<zifuxu<<endl;
          }
          if (zifuxu < 17 && ffff == 1) {
            for (int i = vinCharImg.size() - zifuxu; i < vinCharImg.size();
                 i++) {
              cvReleaseImage(&vinCharImg[i]);
            }
            vinCharImg.erase(vinCharImg.end() - zifuxu, vinCharImg.end());
          }
          if (zifuxu >= 17) //保留接近17个字符图片
          {
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

            if (diji != 0) {
              FindDatePic(src, rect, ifrotate_flag, path_dateimg);
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
//第二次识别//找盖章处参数cishu=1表示第二次查找
bool ExtrVin1(IplImage *src, IplImage *img, CvSVM *svmvin,
              vector<IplImage *> &vinCharImg, const char path_dateimg[]) {
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
  // DeleteRONI(img,720,60,60,20,720,0,20000,0);
  // cvShowImage("sds",img);
  char file_dst[200];      //保存路径
  cvCopy(src, img_Clone1); //归一化后的彩色图像
  cvCopy(img, img_Clone);  //得到的连通区域图片
  int k = 0;               //保存图片用
  bool det = FALSE;        //标记是否检测出正样本
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
      cvResetImageROI(img_Clone1);

      IplImage *img_cc = 0;
      IplImage *xuanzhuan = 0;
      xuanzhuan = fcjiaodu(src, img_c, rect1);
      cvReleaseImage(&img_c); //释放
      img_cc = charprocess0(
          xuanzhuan); // 0表示不保存图片，此函数已对xuanzhuan进行了释放//
      // cvShowImage("aaa",img_cc);
      // cvWaitKey(0);
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
          //获取注册日期区域图片_________________________________________________________________________________////////

          if (diji == 0) {
            FindDatePic(img_Clone1, rect1, 0, path_dateimg);
          }
          //获取注册日期区域图片结束_____________________________________________________________________________///////
          float atio = (float)img_cc->height / img_cc->width;
          IplImage *nimg = cvCreateImage(cvSize(480, 480 * atio), 8, 3);
          cvResize(img_cc, nimg);
          // cvShowImage("hhhhhh",img_cc);
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
          re_cut(matimg, nimg, threshold, sw, ++diji, 1, zifuxu, vinCharImg);
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
            re_cut(matimg, nimg, threshold, sw, diji, 0, zifuxu, vinCharImg);
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
            if (diji != 0) {
              FindDatePic(src, rect1, 0, path_dateimg);
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
  ;
  // cvReleaseImage(&img);
  cvReleaseImage(&img_Clone1);
  return det;
}
//找盖章处参数cishu=1表示第二次查找
IplImage *FindMark(IplImage *src, IplImage *img, int cishu) {
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
    CvRect r = rectangle[0];
    //所确定的目标超过一个
    if (rectangle.size() > 1) {
      int xx = 0, yy = 0;
      for (int i = 1; i < rectangle.size(); i++) {
        CvRect r1 = rectangle[i];
        xx = max(r.x + r.width, r1.x + r1.width);
        yy = max(r.y + r.height, r1.y + r1.height);
        r.x = min(r1.x, r.x);
        r.y = min(r1.y, r.y);
      }
      r.width = xx - r.x;
      r.height = yy - r.y;
    }
    //判断旋转，并且求出盖章区域
    if (r.x < w / 3) {
      if (r.y < h / 3) {
        p3.x = 20;
        p3.y = r.y;
        p4.x = r.x + r.width + 20;
        p4.y = h; ///;-20

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
          p4.y = r.y + r.height; ///-20
          flag = 0;              // if(r.y+r.height<h);
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
          p4.y = h;
        } ///-20
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
        p3.y = r.y - 10; ///
        p4.x = w - 10;
        p4.y = h; //-20
      }

      if (r.height > h / 3) {
        p3.x = 20;
        p3.y = r.y;
        p4.x = w - 20;
        p4.y = r.y + r.height; ///-10
                               // flag=2;
      }
      // flag=2情况
      if (r.y > h * 2 / 3 && (w - (r.x + r.width)) < 4) {
        flag = 2;
        p3.x = r.x - 20;
        p3.y = 0;
        p4.x = r.x + r.width;
        p4.y = r.y + r.height; ///-20
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
    r_img = rotateImage(r_img, 90, FALSE);
  }
  if (flag1 == 0 && flag == 2) {
    r_img = rotateImage(r_img, 90, TRUE);
  }
  if (flag1 == 0 && flag == 3) {
    r_img = rotateImage(r_img, 90, TRUE);
    r_img = rotateImage(r_img, 90, TRUE);
  }
  vector<CvRect>(rectangle).swap(rectangle);
  vector<CvRect>(rectangle0).swap(rectangle0);
  return r_img;
}
//求连通区域图片_(binimg),diji=1或2或3：表示第几次求连通域
void Connected_Domain(IplImage *src, IplImage *binimg, int diji) {
  IplConvKernel *KR1 =
      cvCreateStructuringElementEx(2, 1, 0, 0, CV_SHAPE_RECT); //水平
  IplConvKernel *KR2 =
      cvCreateStructuringElementEx(1, 2, 0, 0, CV_SHAPE_RECT); //垂直
  IplConvKernel *KR3 =
      cvCreateStructuringElementEx(3, 1, 0, 0, CV_SHAPE_RECT); //水平

  IplImage *img1 =
      cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1); //创建灰度图像
  IplImage *img2 =
      cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1); //创建目标图像
  cvCvtColor(src, img1, CV_BGR2GRAY);                 //灰度转换
  cvSmooth(img1, img1, CV_GAUSSIAN, 3, 3);            //高斯滤波
  cvSobel(img1, img2, 0, 1, 3);                       //水平边缘
  cvSobel(img1, img1, 1, 0, 3);                       //垂直边缘

  if (diji == 1) {
    //水平边缘找连通域
    cvThreshold(img1, binimg, 0, 255,
                CV_THRESH_OTSU | CV_THRESH_BINARY); //垂直二值化
    DeleteRONI(binimg, 720, 0, 50, 0, 480, 0, 20000, 0); //删除不满足要求连通域
    //形态学处理_____
    cvDilate(binimg, binimg, KR3, 1);
    cvDilate(binimg, binimg, KR1, 5);
    cvDilate(binimg, binimg, KR1, 5);
    cvErode(binimg, binimg, KR1, 1);
    cvErode(binimg, binimg, KR2, 3);
    cvDilate(binimg, binimg, KR2, 2);
  }
  if (diji == 2) {
    //垂直和水平边缘相加，找盖章处
    cvThreshold(img2, img2, 0, 255,
                CV_THRESH_OTSU | CV_THRESH_BINARY); //水平二值化
    cvThreshold(img1, binimg, 0, 255,
                CV_THRESH_OTSU | CV_THRESH_BINARY); //垂直二值

    //
    plusimg(img2, binimg);
    //边缘相加
  }
  if (diji == 3) {
    //第二次定位找连通域
    cvThreshold(img2, img2, 0, 255,
                CV_THRESH_OTSU | CV_THRESH_BINARY); //水平二值化
    cvDilate(img2, binimg, KR1, 8);
  }
  //内存释放
  cvReleaseImage(&img1);
  cvReleaseImage(&img2);
  cvReleaseStructuringElement(&KR1);
  cvReleaseStructuringElement(&KR2);
  cvReleaseStructuringElement(&KR3);
}
//对原图进行尺度变换，并根据垂直边缘进行旋转
IplImage *resize(IplImage *src) {

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
//提取单个字符的特征矩阵
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
//识别函数
bool recvin(cv::Mat src_mat, char pre[], CvSVM *svmvin, CvSVM *svm33,
            CvSVM *svm35, CvSVM *svmLast5, const char path_failimg[],
            const char path_dateimg[]) {

  string charList = "0123456789ABCDEFGHJKLMNPRSTUVWXYZIQ"; //字符
  IplImage *img = NULL;
  IplImage *src = NULL;
  IplImage ipl_img(src_mat);

  src = &ipl_img;

  img = resize(src);

  // cvReleaseImage(&src);

  IplImage *img1 = NULL;
  img1 = jianqie0(img);
  // cvShowImage("ss",img1);
  cvReleaseImage(&img);

  float hw_atio = (float)img1->height / img1->width;
  IplImage *img2 = cvCreateImage(cvSize(640, 640 * hw_atio), 8, 3);
  cvResize(img1, img2);

  cvReleaseImage(&img1);

  IplImage *binimg0 = cvCreateImage(cvGetSize(img2), 8, 0);
  Connected_Domain(img2, binimg0, 1);

  vector<IplImage *> vinCharImg;
  vinCharImg.clear();
  bool det = false;

  //开始定位Vin and save the  image of date
  cvSaveImage(path_failimg, img2);
  det = ExtrVin(img2, binimg0, svmvin, vinCharImg, path_dateimg);
  cvReleaseImage(&binimg0); //释放

  //第一次定位结束
  if (vinCharImg.size() == 0) {
    det = FALSE;
  }

  IplImage *r_img = NULL;

  if (!det) //第一次定位失败，则开始第二次定位
  {
    // 第二次定位;
    IplImage *binimg1 = cvCreateImage(cvGetSize(img2), 8, 1);

    Connected_Domain(img2, binimg1, 2);

    // cvShowImage("第二次定位找连通域11",binimg1);
    int cishu = 0;
    int ifregion = 1; //标记是否查找vin码区域程
    r_img = FindMark(img2, binimg1, cishu);
    if (r_img == NULL) {
      cishu = 1;
      IplConvKernel *KR1 =
          cvCreateStructuringElementEx(2, 2, 0, 0, CV_SHAPE_RECT);
      cvDilate(binimg1, binimg1, KR1, 4);
      r_img = FindMark(img2, binimg1, cishu);
      cvReleaseStructuringElement(&KR1);
    }
    cvReleaseImage(&binimg1);
    if (r_img != NULL) {
      IplImage *binimg2 = cvCreateImage(cvGetSize(r_img), 8, 1);
      Connected_Domain(r_img, binimg2, 3); // cvShowImage("切割",binimg2);//
      // printf("第二次第一次定位\n");
      vinCharImg.clear();
      cvSaveImage(path_failimg, r_img);
      det = ExtrVin1(r_img, binimg2, svmvin, vinCharImg, path_dateimg);
      cvReleaseImage(&binimg2);
    }
  }
  //______________________________第二次定位结束______________________________________________
  //开始识别;
  cvReleaseImage(&img2);
  int flag = 0;
  // cout<<vinCharImg.size()<<endl;

  if (vinCharImg.size() > 0) {
    CvMat *data_mat;
    data_mat = cvCreateMat(1, 1350, CV_32FC1);
    cvSetZero(data_mat); //初始化为0
    int res;
    int numFu = 0;
    // cout<<vinCharImg.size()<<endl;
    int k = 0;
    for (int charInd = 0; charInd != (int)vinCharImg.size(); charInd++) {

      IplImage *charImg = vinCharImg[charInd];
      if (charImg == NULL) {
        cvReleaseImage(&charImg);
        continue;
      }
      flag = 1;
      FeatureExtraction(charImg, data_mat, 0);

      if (vinCharImg.size() == 17) {
        res = charInd < 12 ? svm33->predict(data_mat)
                           : svmLast5->predict(data_mat);
      } else {
        res = charInd < 12 ? svm35->predict(data_mat)
                           : svmLast5->predict(data_mat);
      }

      if (res >= 33 && numFu < vinCharImg.size() - 17) {
        numFu++;
        cvReleaseImage(&charImg);
        continue;
      }
      //保存识别结果
      pre[k] = (char)charList[res];
      k++;

      cvReleaseImage(&charImg);
    }

    cvReleaseMat(&data_mat);
  }
  //识别不成功
  if (flag == 0) {
    strcpy(pre, "Fail");
    // DateOrFail_image=cv::Mat(r_img,true);
    cvReleaseImage(&r_img);

    return false;
  }
  cvReleaseImage(&r_img);
  return true;
}

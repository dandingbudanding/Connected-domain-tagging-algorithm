#include <iostream>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\core\core.hpp>  
#include <opencv2\imgproc\imgproc.hpp>  
#include <map>  
#include <stack>
#include <string>  
#include <list>    
#include <stdio.h>
#include <time.h>


using namespace std;
using namespace cv;

void myFillSeedFill(const Mat& _binImg, Mat& _labelImg) {
	if (_binImg.empty() || _binImg.type() != CV_8UC1) {
		return;
	}

	_labelImg.release();
	_binImg.convertTo(_labelImg, CV_32SC1);

	int label = 1;
	int rows = _binImg.rows;
	int cols = _binImg.cols;

	vector<pair<int, int>> area;
	int maxLabel = 0;
	int maxLabelNum = 0;

	for (int i = 0; i < rows; ++i) {
		int * p = _labelImg.ptr<int>(i);
		for (int j = 0; j < cols; ++j) {
			if (p[j] == 1) {
				stack<pair<int, int>> s;
				s.push(pair<int, int>(i, j));
				++label;
				int curLabelNum = 0;
				while (!s.empty()) {
					pair<int, int> curPix = s.top();
					int curX = curPix.first;
					int curY = curPix.second;

					int * curxptr = _labelImg.ptr<int>(curX);
					curxptr[curY] = label;

					//_labelImg.at<int>(curX, curY) = label;
					++curLabelNum;
					s.pop();


					if (curX - 1 >= 0) {
						int * prexptr = _labelImg.ptr<int>(curX - 1);
						if (prexptr[curY] == 1) {
							s.push(pair<int, int>(curX - 1, curY));
						}
					}
					if (curY - 1 >= 0) {
						//int * prexptr = _labelImg.ptr<int>(curX-1);
						if (curxptr[curY - 1] == 1) {
							s.push(pair<int, int>(curX, curY - 1));
						}
					}
					if (curX + 1 < rows) {
						int * latexptr = _labelImg.ptr<int>(curX + 1);
						if (latexptr[curY] == 1) {
							s.push(pair<int, int>(curX + 1, curY));
						}
					}
					if (curY + 1 < cols) {
						//int * prexptr = _labelImg.ptr<int>(curX-1);
						if (curxptr[curY + 1] == 1) {
							s.push(pair<int, int>(curX, curY + 1));
						}
					}
				}
				if (curLabelNum > maxLabelNum) {
					maxLabelNum = curLabelNum;
					maxLabel = label;
				}
			}
		}
	}

	for (int i = 0; i < rows; ++i) {
		int * p = _labelImg.ptr<int>(i);
		for (int j = 0; j < cols; ++j) {
			if (p[j] == maxLabel) {
				p[j] = 255;
			}
			else {
				p[j] = 0;
			}
		}
	}

	//总耗时： 0.5秒
}


int father[100000];
vector<int> labelCnt(1000, 0);

void make() {
	for (int i = 0; i < 100000; ++i) {
		father[i] = i;
	}
}

int Find_Set(int x) {
	if (x != father[x]) {
		father[x] = Find_Set(father[x]);
	}
	return father[x];
}

void Union(int x, int y) {
	int fx = Find_Set(x);
	int fy = Find_Set(y);
	if (fx == fy) {
		return;
	}
	father[fy] = fx;
}

void icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg)
{
	// connected component analysis (4-component)  
	// use two-pass algorithm  
	// 1. first pass: label each foreground pixel with a label  
	// 2. second pass: visit each labeled pixel and merge neighbor labels  
	//   
	// foreground pixel: _binImg(x,y) = 1  
	// background pixel: _binImg(x,y) = 0  


	if (_binImg.empty() ||
		_binImg.type() != CV_8UC1)
	{
		return;
	}

	// 1. first pass  

	_lableImg.release();
	_binImg.convertTo(_lableImg, CV_32SC1);

	int label = 1;  // start by 2  
	std::vector<int> labelSet;
	labelSet.push_back(0);   // background: 0  
	labelSet.push_back(1);   // foreground: 1  

	int rows = _binImg.rows;
	int cols = _binImg.cols;
	int* data_firstRows = _lableImg.ptr<int>(0);
	if (data_firstRows[0] == 1) {
		data_firstRows[0] = ++label;
		labelSet.push_back(label);
	}
	for (int j = 1; j < cols; ++j) {  //第一行初始化label
		if (data_firstRows[j] == 1) {
			if (data_firstRows[j - 1] > 0) {
				data_firstRows[j] == data_firstRows[j - 1];
			}
			else {
				data_firstRows[j] = ++label;
				labelSet.push_back(label);
			}
		}
	}
	for (int j = 1; j < rows; ++j) {   //第一列初始化label
		if (_lableImg.at<int>(j, 0) == 1) {
			if (_lableImg.at<int>(j - 1, 0) > 0) {
				_lableImg.at<int>(j, 0) = _lableImg.at<int>(j - 1, 0);
			}
			else {
				_lableImg.at<int>(j, 0) = ++label;
				labelSet.push_back(label);
			}
		}
	}
	//label = 1;
	make();
	for (int i = 1; i < rows; i++)
	{
		int* data_preRow = _lableImg.ptr<int>(i - 1);
		int* data_curRow = _lableImg.ptr<int>(i);
		for (int j = 1; j < cols; j++)
		{
			if (data_curRow[j] == 1)
			{
				std::vector<int> neighborLabels;
				neighborLabels.reserve(2);
				int leftPixel = data_curRow[j - 1];
				int upPixel = data_preRow[j];
				if (leftPixel > 1)
				{
					neighborLabels.push_back(leftPixel);
				}
				if (upPixel > 1)
				{
					neighborLabels.push_back(upPixel);
				}

				if (neighborLabels.empty())
				{
					labelSet.push_back(++label);  // assign to a new label  
					data_curRow[j] = label;
					labelSet[label] = label;
				}
				else
				{
					std::sort(neighborLabels.begin(), neighborLabels.end());
					int smallestLabel = neighborLabels[0];
					data_curRow[j] = smallestLabel;

					for (size_t k = 1; k < neighborLabels.size(); k++) {
						int tmpLabel = neighborLabels[k];

						Union(tmpLabel, smallestLabel);
						Union(smallestLabel, tmpLabel);
					}

				}
			}
		}
	}


	// 2. second pass  
	for (int i = 0; i < rows; i++)
	{
		int* data = _lableImg.ptr<int>(i);
		for (int j = 0; j < cols; j++)
		{
			data[j] = Find_Set(data[j]);
			if (data[j] != 0) {
				labelCnt[data[j]]++;
			}
		}
	}
	vector<int>::iterator maxLabel = max_element(labelCnt.begin(), labelCnt.end());
	cout << *maxLabel << endl;
	int maxL = distance(labelCnt.begin(), maxLabel);
	cout << maxL << endl;

	for (int i = 0; i < rows; i++) {
		int* data = _lableImg.ptr<int>(i);
		for (int j = 0; j < cols; j++) {
			if (data[j] == maxL) {
				data[j] = 255;
			}
			else {
				data[j] = 0;
			}
		}
	}

}


cv::Scalar icvprGetRandomColor()
{
	uchar r = 255 * (rand() / (1.0 + RAND_MAX));
	uchar g = 255 * (rand() / (1.0 + RAND_MAX));
	uchar b = 255 * (rand() / (1.0 + RAND_MAX));
	return cv::Scalar(b, g, r);
}


void icvprLabelColor(const cv::Mat& _labelImg, cv::Mat& _colorLabelImg)
{
	if (_labelImg.empty() ||
		_labelImg.type() != CV_32SC1)
	{
		return;
	}

	std::map<int, cv::Scalar> colors;

	int rows = _labelImg.rows;
	int cols = _labelImg.cols;

	_colorLabelImg.release();
	_colorLabelImg.create(rows, cols, CV_8UC3);
	_colorLabelImg = cv::Scalar::all(0);

	for (int i = 0; i < rows; i++)
	{
		const int* data_src = (int*)_labelImg.ptr<int>(i);
		uchar* data_dst = _colorLabelImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			int pixelValue = data_src[j];
			if (pixelValue > 1)
			{
				if (colors.count(pixelValue) <= 0)
				{
					colors[pixelValue] = icvprGetRandomColor();
				}
				cv::Scalar color = colors[pixelValue];
				*data_dst++ = color[0];
				*data_dst++ = color[1];
				*data_dst++ = color[2];
			}
			else
			{
				data_dst++;
				data_dst++;
				data_dst++;
			}
		}
	}
}

int main()
{
	time_t t_start, t_end;
	/* Mat img, res;  */
	Mat binImage = imread("1.bmp", 0);    //将读入的彩色图像直接以灰度图像读入
	//Mat binImage = imread("liantongyu.jpg",0);   //小图，在递归调用中可用，图太大则只能使用TwoPass方法
	//Mat binImage = imread("connect2.jpg",0);  
	Mat showimg;


	showimg.release();
	binImage.convertTo(showimg, CV_8UC1);
	for (int i = 0; i < showimg.rows; ++i) {
		uchar * p = showimg.ptr<uchar>(i);
		for (int j = 0; j < showimg.cols; ++j) {
			if (p[j] > 50)
				p[j] = 225;
			else
				p[j] = 0;
		}
	}

	cv::imshow("showing", showimg);

	//cv::Mat binImage = cv::imread("../icvpr.com.jpg", 0) ;  
	//cv::threshold(binImage, binImage, 50, 1, CV_THRESH_BINARY_INV) ;  
	for (int i = 0; i < binImage.rows; ++i) {
		uchar * p = binImage.ptr<uchar>(i);
		for (int j = 0; j < binImage.cols; ++j) {
			if (p[j] > 50)
				p[j] = 1;
			else
				p[j] = 0;
		}
	}


	// connected component labeling  
	cv::Mat labelImg;

	t_start = clock();
	//icvprCcaByTwoPass(binImage, labelImg) ;         // cost 0.184s
	myFillSeedFill(binImage, labelImg);             // cost 0.526s


	t_end = clock();
	double duration = (double)(t_end - t_start) / CLOCKS_PER_SEC;
	cout << "cost " << duration << " s" << endl;



	// show result  
	cv::Mat grayImg;
	//labelImg *= 10 ;  
	labelImg.convertTo(grayImg, CV_8UC1);
	cv::imshow("labelImg", grayImg);

	cv::Mat colorLabelImg;
	icvprLabelColor(labelImg, colorLabelImg);
	cv::imshow("colorImg", colorLabelImg);
	cv::waitKey(0);

	return 0;
}

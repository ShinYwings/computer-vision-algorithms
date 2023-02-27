#include <cstdio>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>

///////////////////////////////////////////////////////////////////////////////

using namespace std;
using namespace cv;

vector<cv::Mat> start;
vector<cv::Mat> finish;

void LowpassFilter (const cv::Mat &image, cv::Mat &result)
{
    cv::Mat kernel (3, 3, CV_32F, cv::Scalar (0));

    kernel.at <float> (0, 0) = kernel.at <float> (2, 0) = 1.0f / 16.0f;
    kernel.at <float> (0, 2) = kernel.at <float> (2, 2) = 1.0f / 16.0f;
    kernel.at <float> (1, 0) = kernel.at <float> (1, 2) = 1.0f /  8.0f;
    kernel.at <float> (0, 1) = kernel.at <float> (2, 1) = 1.0f /  8.0f;
    kernel.at <float> (1, 1) = 1.0f /  4.0f;

    cv::filter2D (image, result, image.depth (), kernel);
}

uchar get_value (cv::Mat const &image, cv::Point2f index)
{
    if (index.x >= image.rows)   index.x = image.rows - 1.0f;
    else if (index.x < 0)         index.x = .0f;

    if (index.y >= image.cols)    index.y = image.cols - 1.0f;
    else if (index.y < 0)         index.y = .0f;

    return image.at <uchar> (index.x, index.y);
}

uchar get_subpixel_value (cv::Mat const &image, cv::Point2f index)
{
    float floorX = (float) floor (index.x);
    float floorY = (float) floor (index.y);

    float fractX = index.x - floorX;
    float fractY = index.y - floorY;

    return ((1.0f - fractX) * (1.0f - fractY) * get_value (image, cv::Point2f (floorX, floorY))
                + (fractX * (1.0f - fractY) * get_value (image, cv::Point2f (floorX + 1.0f, floorY)))
                + ((1.0f - fractX) * fractY * get_value (image, cv::Point2f (floorX, floorY + 1.0f)))
                + (fractX * fractY * get_value (image, cv::Point2f (floorX + 1.0f, floorY + 1.0f))));
}

void BuildPyramid (cv::Mat const &input, std::vector <cv::Mat> &outputArray, int maxLevel)
{
    outputArray.push_back (input);
    for (int k = 1; k <= maxLevel; ++k)
    {
        cv::Mat prevImage;
        LowpassFilter (outputArray.at (k - 1), prevImage);  // Image Blurring

        int limRows = (prevImage.rows + 1) / 2;
        int limCols = (prevImage.cols + 1) / 2;

        cv::Mat currMat (limRows, limCols, CV_8UC1, cv::Scalar (0));

        for (int i = 0; i < limRows; ++i)
        {
            for (int j = 0; j < limCols; ++j)
            {
                /// Index always integer
                float indexX = 2 * i;
                float indexY = 2 * j;

                float firstSum = (get_value (prevImage, cv::Point2f (indexX, indexY))) / 4.0f;

                float secondSum = .0f;
                secondSum += get_value (prevImage, cv::Point2f (indexX - 1.0f, indexY));
                secondSum += get_value (prevImage, cv::Point2f (indexX + 1.0f, indexY));
                secondSum += get_value (prevImage, cv::Point2f (indexX, indexY - 1.0f));
                secondSum += get_value (prevImage, cv::Point2f (indexX, indexY + 1.0f));
                secondSum /= 8.0f;

                float thirdSum = .0f;
                thirdSum += get_value (prevImage, cv::Point2f (indexX - 1.0f, indexY - 1.0f));
                thirdSum += get_value (prevImage, cv::Point2f (indexX + 1.0f, indexY - 1.0f));
                thirdSum += get_value (prevImage, cv::Point2f (indexX - 1.0f, indexY + 1.0f));
                thirdSum += get_value (prevImage, cv::Point2f (indexX + 1.0f, indexY + 1.0f));
                thirdSum /= 16.0f;

                currMat.at <uchar> (i, j) = firstSum + secondSum + thirdSum;
            }
        }
        outputArray.push_back (currMat);
    }
}

int LucasKanade (std::vector <cv::Mat> &prevImage, std::vector <cv::Mat> &nextImage,
                   cv::Mat &prevPoint, cv::Mat &nextPoint)
{
    cv::Mat piramidalGuess (2, 1, CV_32F, cv::Scalar (0));
    cv::Mat opticalFlowFinal (2, 1, CV_32F, cv::Scalar (0));

    for (int level = prevImage.size () - 1; level >= 0; --level)
    {
        cv::Mat currPoint (2, 1, CV_32F, cv::Scalar (0));
        currPoint.at <float> (0, 0) =  prevPoint.at <float> (0, 0) / pow (2, level);
        currPoint.at <float> (1, 0) =  prevPoint.at <float> (1, 0) / pow (2, level);

        int omegaX = 7;
        int omegaY = 7;

        /// Define the area
        float indexXLeft = currPoint.at <float> (0, 0) - omegaX;
        float indexYLeft = currPoint.at <float> (1, 0) - omegaY;

        float indexXRight = currPoint.at <float> (0, 0) + omegaX;
        float indexYRight = currPoint.at <float> (1, 0) + omegaY;

        cv::Mat gradient (2, 2, CV_32F, cv::Scalar (0));
        std::vector <cv::Point2f> derivatives;

        for (float i = indexXLeft; i <= indexXRight; i += 1.0f)
        {
            for (float j = indexYLeft; j <= indexYRight; j += 1.0f)
            {
                float derivativeX = ( get_subpixel_value (prevImage.at (level), cv::Point2f (i + 1.0f, j))
                                        - get_subpixel_value (prevImage.at (level), cv::Point2f (i - 1.0f, j))) / 2.0f;


                float derivativeY = ( get_subpixel_value (prevImage.at (level), cv::Point2f (i, j + 1.0f))
                                    - get_subpixel_value (prevImage.at (level), cv::Point2f (i, j - 1.0f))) / 2.0f;

                derivatives.push_back (cv::Point2f (derivativeX, derivativeY));

                gradient.at <float> (0, 0) += derivativeX * derivativeX;
                gradient.at <float> (0, 1) += derivativeX * derivativeY;
                gradient.at <float> (1, 0) += derivativeX * derivativeY;
                gradient.at <float> (1, 1) += derivativeY * derivativeY;
            }
        }

        gradient = gradient.inv ();

        int maxCount = 3;
        cv::Mat opticalFlow (2, 1, CV_32F, cv::Scalar (0));
        for (int k = 0; k < maxCount; ++k)
        {
            int cnt = 0;
            cv::Mat imageMismatch (2, 1, CV_32F, cv::Scalar (0));
            for (float i = indexXLeft; i <= indexXRight; i += 1.0f)
            {
                for (float j = indexYLeft; j <= indexYRight; j += 1.0f)
                {
                    float nextIndexX = i + piramidalGuess.at <float> (0, 0) + opticalFlow.at <float> (0, 0);
                    float nextIndexY = j + piramidalGuess.at <float> (1, 0) + opticalFlow.at <float> (1, 0);

                    int pixelDifference = (int) ( get_subpixel_value (prevImage.at (level), cv::Point2f (i, j))
                                                - get_subpixel_value (nextImage.at (level), cv::Point2f (nextIndexX, nextIndexY)));

                    imageMismatch.at <float> (0, 0) += pixelDifference * derivatives.at (cnt).x;
                    imageMismatch.at <float> (1, 0) += pixelDifference * derivatives.at (cnt).y;

                    cnt++;
                }
            }

            opticalFlow += gradient * imageMismatch;
        }

        if (level == 0)     opticalFlowFinal = opticalFlow;
        else
            piramidalGuess = 2 * (piramidalGuess + opticalFlow);

    }

    opticalFlowFinal += piramidalGuess;
    nextPoint = prevPoint + opticalFlowFinal;

    if (    (nextPoint.at <float> (0, 0) < 0) || (nextPoint.at <float> (1, 0) < 0)  ||
            (nextPoint.at <float> (0, 0) >= prevImage.at (0).rows)                  ||
            (nextPoint.at <float> (1, 0) >= prevImage.at (0).cols))
    {
        printf ("Object is lost\n");
        return 0;
    }

 //   printf ("Curr point: %fx%f\n", prevPoint.at <float> (0, 0), prevPoint.at <float> (1, 0));
 //   printf ("New point: %fx%f\n", nextPoint.at <float> (0, 0), nextPoint.at <float> (1, 0));

    return 1;
}

///////////////////////////////////////////////////////////////////////////////

int main (int argc, char **argv)
{
   	if (argc != 2)
    {
        printf ("Add video parameters\n");
		return -1;
	}

    cv::VideoCapture capture (argv [1]);

	if (!capture.isOpened ())
    {
		printf ("Can't load %s image file\n", argv [1]);
		return -1;
	}

	double rate = capture.get( CAP_PROP_FPS );

	bool stop (false);
	cv::Mat frame, gray, grayPrev, prevFrame;
	cv::namedWindow( "Extracted Frame" );
	int delay = 1000 / rate;

    //VideoWriter
    Size S = Size((int) capture.get(CAP_PROP_FRAME_WIDTH),
                  (int) capture.get(CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter outputVideo("output.MP4", VideoWriter::fourcc('D','I','V','X'), rate, S, true);

    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << endl;
        return -1;
    }

    capture.read (prevFrame);
    cv::cvtColor (prevFrame, grayPrev, COLOR_BGR2GRAY);
    
    cv::Mat mask (S, CV_8UC3, Scalar(0,0,0)); 
    cv::Mat res (S, CV_8UC3, Scalar(0,0,0)); 

    for(int i = 50; i < prevFrame.cols -50; i=i+50)
    {
        for(int j = 50; j < prevFrame.rows -50; j=j+50)
        {   
            cv::Mat startPnt (2, 1, CV_32F, cv::Scalar (0));
            startPnt.at <float> (0, 0) = (float) j;
            startPnt.at <float> (1, 0) = (float) i;
            // cv::circle(prevFrame, cv::Point ((int) startPnt.at <float> (1, 0), (int) startPnt.at <float> (0, 0)), 7, cv::Scalar (255, 0, 0));
            start.push_back(startPnt);
        }
    }

	while ( !stop )
    {
        std::vector <cv::Mat> output1;
        std::vector <cv::Mat> output2;

		if (!capture.read (frame )) break;

        cv::cvtColor (frame, gray, COLOR_BGR2GRAY);

        BuildPyramid (grayPrev, output1, 4);
        BuildPyramid (gray, output2, 4);

        for(int i=0; i<start.size() ; i++)
        {
            cv::Mat finishPnt  (2, 1, CV_32F, cv::Scalar (0));
            LucasKanade (output1, output2, start[i], finishPnt);
            finish.push_back(finishPnt);

            cv::Point start_xy = cv::Point ((int) start[i].at <float> (1, 0), (int) start[i].at <float> (0, 0));
            cv::Point finish_xy = cv::Point ((int) finish[i].at <float> (1, 0), (int) finish[i].at <float> (0, 0));
            
            // gradient direction
            cv::arrowedLine (frame, start_xy, finish_xy, cv::Scalar (0, 0, 255), 1);

        }
        
        res = frame + mask;
        imshow ("Extracted Frame", res);
        outputVideo << res;

        gray.copyTo (grayPrev);

        for(int i =0; i<4; i++)
        {
            finish[i].copyTo (start[i]);

        }
        finish.clear();

		if (cv::waitKey (delay) >= 0) stop = true;
	}
	capture.release();
    outputVideo.release();

    cv::waitKey ();

    return 0;
}
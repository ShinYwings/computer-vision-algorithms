
```cpp
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
    return 1;
}
```
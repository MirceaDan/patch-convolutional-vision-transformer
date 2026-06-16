#ifndef CAMERA_H
#define CAMERA_H

#include <libcamera/libcamera.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

class RaspiCamera {
public:
    RaspiCamera();
    ~RaspiCamera();

    bool start();
    bool getFrame(cv::Mat& frame);
    void stop();

private:
    std::unique_ptr<libcamera::CameraManager> cm_;
    std::shared_ptr<libcamera::Camera> camera_;
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;

    libcamera::Stream* stream_;
    std::vector<std::unique_ptr<libcamera::Request>> requests_;

    int width_;
    int height_;
    int stride_;
};

#endif

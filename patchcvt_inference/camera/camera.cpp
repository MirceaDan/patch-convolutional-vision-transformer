#include "camera.h"

#include <iostream>
#include <sys/mman.h>

using namespace libcamera;

RaspiCamera::RaspiCamera()
    : stream_(nullptr),
      width_(0),
      height_(0),
      stride_(0)
{
}

RaspiCamera::~RaspiCamera() {
    stop();
}

bool RaspiCamera::start() {
    cm_ = std::make_unique<CameraManager>();
    if (cm_->start()) {
        std::cerr << "Failed to start CameraManager\n";
        return false;
    }

    if (cm_->cameras().empty()) {
        std::cerr << "No cameras found\n";
        return false;
    }

    camera_ = cm_->cameras()[0];
    if (camera_->acquire()) {
        std::cerr << "Failed to acquire camera\n";
        return false;
    }

    // EXPLICIT ISP
    auto config = camera_->generateConfiguration({ StreamRole::Viewfinder });
    auto& cfg = config->at(0);

    cfg.pixelFormat = formats::RGB888;
    cfg.size = {1536, 864};
    cfg.bufferCount = 2;

    if (config->validate() == CameraConfiguration::Invalid) {
        std::cerr << "Invalid camera configuration\n";
        return false;
    }

    if (camera_->configure(config.get())) {
        std::cerr << "Failed to configure camera\n";
        return false;
    }

    stream_ = cfg.stream();
    width_  = cfg.size.width;
    height_ = cfg.size.height;
    stride_ = cfg.stride;

    allocator_ = std::make_unique<FrameBufferAllocator>(camera_);
    if (allocator_->allocate(stream_) < 0) {
        std::cerr << "Failed to allocate buffers\n";
        return false;
    }

    for (auto& buffer : allocator_->buffers(stream_)) {
        auto req = camera_->createRequest();
        if (!req) {
            std::cerr << "Failed to create request\n";
            return false;
        }

        if (req->addBuffer(stream_, buffer.get())) {
            std::cerr << "Failed to add buffer to request\n";
            return false;
        }

        requests_.push_back(std::move(req));
    }

    // CONTROLS — Python equivalent
    ControlList controls;
    controls.set(controls::ExposureTime, 200000);     // 200 ms
    controls.set(controls::AnalogueGain, 8.0f);
    controls.set(controls::AwbEnable, false);
    controls.set(libcamera::controls::draft::NoiseReductionMode, 2);

    if (camera_->start(&controls)) {
        std::cerr << "Failed to start camera\n";
        return false;
    }

    for (auto& req : requests_)
        camera_->queueRequest(req.get());

    return true;
}

bool RaspiCamera::getFrame(cv::Mat& frame) {
    Request* latest = nullptr;

    for (auto& req : requests_) {
        if (req->status() == Request::RequestComplete) {
            latest = req.get();
        }
    }

    if (!latest)
        return false;
    
    for (auto& req : requests_) {
        if (req.get() != latest && req->status() == Request::RequestComplete) 
        {
            req->reuse(Request::ReuseBuffers);
            camera_->queueRequest(req.get());
        }
    }

    auto buffer = latest->buffers().begin()->second;
    const FrameBuffer::Plane& plane = buffer->planes()[0];

    void* mem = mmap(nullptr, plane.length,
                     PROT_READ, MAP_SHARED,
                     plane.fd.get(), 0);

    if (mem == MAP_FAILED) {
        std::cerr << "mmap failed\n";
        return false;
    }

    cv::Mat tmp(height_, width_, CV_8UC3, mem, stride_);
    frame = tmp.clone();

    munmap(mem, plane.length);

    latest->reuse(Request::ReuseBuffers);
    camera_->queueRequest(latest);

    return true;
}

void RaspiCamera::stop() {
    if (camera_) {
        camera_->stop();
        camera_->release();
        camera_.reset();
    }

    if (cm_) {
        cm_->stop();
        cm_.reset();
    }
}

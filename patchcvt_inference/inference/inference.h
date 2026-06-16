#ifndef INFERENCE_H_
#define INFERENCE_H_

#include <cmath>
#include <iostream>
#include <fstream>
#include <jsoncpp/json/json.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

using namespace std;

struct ModelInfo
{
    std::string model_name;
    std::string format;
    bool quantized = true;

    int input_width;
    int input_height;
    int input_channels;
    std::string input_layout;
    std::string input_dtype;

    bool normalization_enabled;
    std::vector<float> mean;
    std::vector<float> std;

    std::vector<std::string> classes;

    std::string output_type;
    bool apply_softmax;
};

class InferenceEngine {
public:
    bool loadModel(const std::string &modelPath);
    std::vector<float> softmax(const std::vector<float>& logits);
    std::vector<float> run(const cv::Mat& frame);
    void loadModelInfo(const std::string &path);
    
    ModelInfo modelInfo;

private:
    torch::jit::Module model;
};

#endif
#include "inference.h"

bool InferenceEngine::loadModel(const std::string &modelPath)
{
    try {
        model = torch::jit::load(modelPath, torch::kCPU);
        model.eval();
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<float> InferenceEngine::softmax(const std::vector<float>& logits)
{
    std::vector<float> result(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        result[i] = std::exp(logits[i] - max_logit);
        sum += result[i];
    }

    for (float& v : result)
        v /= sum;

    return result;
}

vector<float> InferenceEngine::run(const cv::Mat& frame)
{
    if (frame.empty())
        throw std::runtime_error("Empty frame");

    // ---------------- PREPROCESS ----------------
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(224, 224));

    cv::Mat img_f32;
    resized.convertTo(img_f32, CV_32F, 1.0 / 255.0);

    // ---------------- TO TENSOR ----------------
    auto inputTensor = torch::from_blob(
        img_f32.data,
        {1, img_f32.rows, img_f32.cols, 3},
        torch::kFloat32
    ).clone();

    // NHWC → NCHW
    inputTensor = inputTensor.permute({0, 3, 1, 2});

    // ---------------- NORMALIZATION ----------------
    const std::vector<double> mean = {0.485, 0.456, 0.406};
    const std::vector<double> std  = {0.229, 0.224, 0.225};

    for (int c = 0; c < 3; ++c)
    {
        inputTensor[0][c] = inputTensor[0][c].sub(mean[c]).div(std[c]);
    }

    // ---------------- FORWARD ----------------
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputTensor);

    torch::Tensor output = model.forward(inputs).toTensor();
    output = output.squeeze(0).to(torch::kCPU);

    // ---------------- OUTPUT ----------------
    std::vector<float> logits(2);
    auto acc = output.accessor<float,1>();

    for (int i = 0; i < 2; ++i)
        logits[i] = acc[i];

    return softmax(logits);
}

void InferenceEngine::loadModelInfo(const std::string &path) 
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open model_info.json");

    Json::Value root;
    file >> root;

    ModelInfo info;

    info.model_name = root["model_name"].asString();
    info.format     = root["format"].asString();
    info.quantized  = root["quantized"].asBool();

    const auto &input = root["input"];
    info.input_width    = input["width"].asInt();
    info.input_height   = input["height"].asInt();
    info.input_channels = input["channels"].asInt();
    info.input_layout   = input["layout"].asString();
    info.input_dtype    = input["dtype"].asString();

    const auto &norm = root["normalization"];
    info.normalization_enabled = norm["enabled"].asBool();

    info.mean.clear();
    info.std.clear();
    for (const auto &v : norm["mean"])
        info.mean.push_back(v.asFloat());
    for (const auto &v : norm["std"])
        info.std.push_back(v.asFloat());

    info.classes.clear();
    for (const auto &c : root["classes"])
        info.classes.push_back(c.asString());

    const auto &output = root["output"];
    info.output_type   = output["type"].asString();
    info.apply_softmax = output["apply_softmax"].asBool();

    modelInfo = info;
}

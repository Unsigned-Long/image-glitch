#include "glitch.h"

namespace ns_glitch {
  bool GlitchSolver::solve(const cv::Mat &colorImg, cv::Mat &dst, const std::vector<const MetaGlitch *> &glitches) {
    CV_Assert(colorImg.type() == CV_8UC3);
    colorImg.copyTo(dst);

    bool state = true;

    for (const auto &glitch : glitches) {
      if (!glitch->operator()(dst)) {
        state = false;
        break;
      }
    }
    return state;
  }

  bool RGBSplitGlitch::operator()(cv::Mat &img) const {
    cv::Mat bgr[3];
    cv::split(img, bgr);
    auto timeCount = std::chrono::system_clock::now().time_since_epoch().count();
    auto sinVal = std::sin(timeCount);
    float radius_noice = 2.0f * sinVal * (sinVal > 0.8) * radius + noice(engine);

    float deltaX = std::cos(theta) * radius_noice;
    float deltaY = std::sin(theta) * radius_noice;
    cv::Mat trans = (cv::Mat_<float>(2, 3) << 1, 0, deltaX, 0, 1, deltaY);
    cv::warpAffine(bgr[0], bgr[0], trans, img.size(), 1, cv::BorderTypes::BORDER_REFLECT101);
    cv::warpAffine(bgr[2], bgr[2], trans * -1, img.size(), 1, cv::BorderTypes::BORDER_REFLECT101);
    cv::merge(bgr, 3, img);
    return true;
  }

  bool ImageBlockGlitch::operator()(cv::Mat &img) const {
    int blockRows = img.rows / bh;
    int blockCols = img.cols / bw;
    for (int i = 0; i != blockRows; ++i) {
      for (int j = 0; j != blockCols; ++j) {
        float strength = noice(engine);
        if (strength < 0.8f) {
          continue;
        }
        cv::Range rowRange(i * bh, std::min((i + 1) * bh, img.rows));
        cv::Range colRange(j * bw, std::min((j + 1) * bw, img.cols));
        cv::Mat block = img(rowRange, colRange);
        rgbGlitch(block);
      }
    }
    return true;
  }
} // namespace ns_glitch

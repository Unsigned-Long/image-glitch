#ifndef GLITCH_H
#define GLITCH_H

#include "artwork/logger/logger.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <random>

namespace ns_glitch {
  class MetaGlitch {
  protected:
    mutable std::default_random_engine engine;

  public:
    MetaGlitch() = default;

  public:
    virtual bool operator()(cv::Mat &img) const = 0;
  };

  class GlitchSolver {
  public:
    GlitchSolver() = default;

  public:
    bool solve(const cv::Mat &colorImg, cv::Mat &dst, const std::vector<const MetaGlitch *> &glitches);
  };

  class RGBSplitGlitch : public MetaGlitch {
  private:
    float theta;
    float radius;
    mutable std::normal_distribution<float> noice;

  public:
    RGBSplitGlitch(int deltaX, int deltaY) : noice(0.0f, 10.0f) {
      theta = std::atan2(deltaY, deltaX);
      radius = std::sqrt(deltaX * deltaX + deltaY + deltaY);
    };

  public:
    virtual bool operator()(cv::Mat &img) const override;
  };

  class ImageBlockGlitch : public MetaGlitch {
  private:
    int bw;
    int bh;
    int deltaX;
    int deltaY;
    mutable std::uniform_real_distribution<float> noice;
    RGBSplitGlitch rgbGlitch;

  public:
    ImageBlockGlitch(int blockWidth, int blockHeight, int deltaX, int deltaY)
        : bw(blockWidth), bh(blockHeight), deltaX(deltaX), deltaY(deltaY),
          noice(0.0f, 1.0f), rgbGlitch(deltaX, deltaY){};

  public:
    virtual bool operator()(cv::Mat &img) const override;
  };
} // namespace ns_glitch

#endif
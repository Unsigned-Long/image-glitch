#include "glitch.h"

int main(int argc, char const *argv[]) {
  cv::Mat img = cv::imread("../img/img.png", cv::IMREAD_COLOR);

  ns_glitch::RGBSplitGlitch rgb(50, 50);
  ns_glitch::ImageBlockGlitch imgBlock(120, 100, 20, 15);
  ns_glitch::GlitchSolver solver;

  cv::Mat dst;
  cv::namedWindow("win", cv::WINDOW_FREERATIO);
  while (true) {
    if (!solver.solve(img, dst, {&imgBlock})) {
      continue;
    }
    cv::imshow("win", dst);
    cv::waitKey(100);
  }

  return 0;
}

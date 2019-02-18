using namespace std;

#include "util/MatrixInverter.h"

namespace ldso {
	namespace util {

		MatXX MatrixInverter::invertPosDef(MatXX M, bool fast)
		{
			if (fast)
			{
				VecX SVec = (M.diagonal().cwiseAbs() + VecX::Constant(M.cols(), 10)).cwiseSqrt();
				VecX SVecI = SVec.cwiseInverse();
				MatXX MScaled = MatXX::Zero(M.rows(), M.cols());
				MScaled.triangularView<Eigen::Upper>() = SVecI.asDiagonal() * M.selfadjointView<Eigen::Upper>().toDenseMatrix() * SVecI.asDiagonal();

				MatXX res = MatXX::Zero(M.rows(), M.cols());
				res.triangularView<Eigen::Upper>() = SVecI.asDiagonal() * MScaled.selfadjointView<Eigen::Upper>().ldlt().solve(MatXX::Identity(M.rows(), M.cols())) * SVecI.asDiagonal();

				//LOG(INFO) << res.selfadjointView<Eigen::Upper>().toDenseMatrix().eigenvalues();
				//LOG(INFO) << M.selfadjointView<Eigen::Upper>().toDenseMatrix().inverse().eigenvalues();

				return res;
			}
			else
			{
				VecX SVec = (M.diagonal().cwiseAbs() + VecX::Constant(M.cols(), 10)).cwiseSqrt();
				VecX SVecI = SVec.cwiseInverse();
				MatXX MScaled = MatXX::Zero(M.rows(), M.cols());
				MScaled.triangularView<Eigen::Upper>() = SVecI.asDiagonal() * M.selfadjointView<Eigen::Upper>().toDenseMatrix() * SVecI.asDiagonal();
				Eigen::JacobiSVD<MatXX> svd(MScaled.selfadjointView<Eigen::Upper>(), Eigen::ComputeThinU | Eigen::ComputeThinV);
				MatXX res = MatXX::Zero(M.rows(), M.cols());

				VecX dinv = svd.singularValues();

				for (int i = 0; i < dinv.size(); i++) {
					if (dinv[i] <= 0) {
						dinv[i] = 0;
					}
					else {
						dinv[i] = 1 / dinv[i];
					}
				}

				res.triangularView<Eigen::Upper>() = SVecI.asDiagonal() * svd.matrixV() * dinv.asDiagonal() * svd.matrixU().transpose() * SVecI.asDiagonal();

				//LOG(INFO) << res.selfadjointView<Eigen::Upper>().toDenseMatrix().eigenvalues();
				//LOG(INFO) << M.selfadjointView<Eigen::Upper>().toDenseMatrix().inverse().eigenvalues();

				return res;
			}
		}
	}
}

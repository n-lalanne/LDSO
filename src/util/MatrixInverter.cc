using namespace std;

#include "util/MatrixInverter.h"

namespace ldso {
	namespace util {

		MatXX MatrixInverter::invertPosDef(MatXX M)
		{
			if (M.determinant() < 1e-10)
				M += VecX::Constant(M.cols(), 1e-10).asDiagonal();
			VecX SVec = (M.diagonal().cwiseAbs() + VecX::Constant(M.cols(), 10)).cwiseSqrt();
			VecX SVecI = SVec.cwiseInverse();
			MatXX MScaled =  M ;
			MScaled = 0.5f * (MScaled + MScaled.transpose());
			MScaled = MScaled.inverse();
			//MScaled = SVec.asDiagonal() * MScaled * SVec.asDiagonal();
			return 0.5 * (MScaled + MScaled.transpose());
		}
	}
}

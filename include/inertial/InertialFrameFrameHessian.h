#pragma once
#ifndef VIDSO_INERTIAL_FRAME_FRAME_HESSIAN_H_
#define VIDSO_INERTIAL_FRAME_FRAME_HESSIAN_H_

using namespace std;

#include "NumTypes.h"
#include "inertial/InertialFrameHessian.h"
#include "inertial/PreIntegration.h"

namespace ldso {
	namespace inertial {
		class InertialFrameHessian;

		class InertialFrameFrameHessian {
		public:
			InertialFrameFrameHessian(shared_ptr<inertial::PreIntegration> preIntegration);
			void linearize(double visualWeight);
			static void computeJacobianAndResidual(Vec15 &r, Mat1515 &J_from, Mat1515 &J_to, shared_ptr<inertial::PreIntegration> preIntegration, Vec3 pi, Vec3 pj, SO3 Riw, SO3 Rjw, SO3 Rwj, Vec3 vi, Vec3 vj, Vec3 bgi, Vec3 bgj, Vec3 bai, Vec3 baj);
			static void computeResidual(Vec15 &r, shared_ptr<inertial::PreIntegration> preIntegration, Vec3 pi, Vec3 pj, SO3 Riw, SO3 Rjw, SO3 Rwj, Vec3 vi, Vec3 vj, Vec3 bgi, Vec3 bgj, Vec3 bai, Vec3 baj);
			// start at this frame, i
			shared_ptr<InertialFrameHessian> from;
			// go to this frame, j
			shared_ptr<InertialFrameHessian> to;
			
			// H: start at this frame, i
			Mat1515 H_from;
			// b: start at this frame, i
			Vec15 b_from;

			// H: go to this frame, j
			Mat1515 H_to;
			// b: go to this frame, j
			Vec15 b_to;

			//0-2: r_R, 3-5: r_v, 6-8: r_p, 9-11: r_bg, 12-14: r_ba
			Vec15 r;

			//0-2: r_R, 3-5: r_v, 6-8: r_p, 9-11: r_bg, 12-14: r_ba
			Mat1515 J_from;
			Mat1515 J_to;

			double energy;
			shared_ptr<inertial::PreIntegration> preIntegration;
		private:
			
		};
	}
}

#endif// VIDSO_INERTIAL_FRAME_FRAME_HESSIAN_H_
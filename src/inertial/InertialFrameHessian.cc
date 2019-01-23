using namespace std;

#include "inertial/InertialFrameHessian.h"
#include "inertial/InertialUtility.h"

namespace ldso {
	namespace inertial {
		void InertialFrameHessian::linearize(shared_ptr<InertialHessian> inertialHessian)
		{
			H.setZero();
			b.setZero();
			J.setZero();
			r.setZero();

			if (from)
			{
				from->linearize();
				H.block<15, 15>(10, 10) += from->H_from;
				b.block<15, 1>(10, 0) += from->b_from;
			}

			if (to)
			{
				H.block<15, 15>(10, 10) += to->H_to;
				b.block<15, 1>(10, 0) += to->b_to;
			}

			Vec3 dr_p_ds = -exp(inertialHessian->scale_PRE)*(inertialHessian->R_WD_PRE*(fh->PRE_camToWorld * inertialHessian->T_CB.translation()));
			Mat33 dr_p_da1 = exp(inertialHessian->scale_PRE)*inertialHessian->R_WD_PRE.matrix();
			Mat33 dr_p_dq = dr_p_da1 * fh->PRE_camToWorld.so3().matrix();


			r.block<3, 1>(0, 0) = (T_WB_PRE.so3()*inertialHessian->T_BC.so3()*fh->PRE_worldToCam.so3()*inertialHessian->R_DW_PRE).log();
			r.block<3, 1>(3, 0) = T_WB_PRE.translation() + dr_p_ds;

			Mat33 JrInv = InertialUtility::JrInv(r.block<3, 1>(0, 0));

			Mat33 dr_R_dphi = inertialHessian->R_WD_PRE.matrix() * fh->PRE_camToWorld.so3().matrix();
			Mat33 dr_R_dw = dr_R_dphi * (inertialHessian->T_CB.so3() * T_BW_PRE.so3()).matrix();

			////dr_R_dq
			//J.block<3, 3>(0, 0) = Mat33::Zero();
			//dr_R_dphi
			J.block<3, 3>(0, 3) = JrInv * dr_R_dphi;
			//dr_R_da
			J.block<3, 3>(0, 6) = JrInv * inertialHessian->R_WD_PRE.matrix();
			////dr_R_ds
			//J.block<3, 1>(0, 9) = Vec3::Zero();
			////dr_R_du
			//J.block<3, 3>(0, 10) = Mat33::Zero();
			//dr_R_dw
			J.block<3, 3>(0, 13) = JrInv * dr_R_dw;


			//dr_p_dq
			J.block<3, 3>(3, 0) = dr_p_dq;
			//dr_p_dphi
			J.block<3, 3>(3, 3) = -dr_p_dq * SO3::hat(inertialHessian->T_CB.translation());
			//dr_p_da
			J.block<3, 3>(3, 6) = -dr_p_da1 * SO3::hat(fh->PRE_camToWorld * inertialHessian->T_CB.translation());
			//dr_p_ds
			J.block<3, 1>(3, 9) = dr_p_ds;
			//dr_p_du
			J.block<3, 3>(3, 10) = Mat33::Identity();
			//dr_p_dw
			J.block<3, 3>(3, 13) = -SO3::hat(T_WB_PRE.translation());


			//J.block<9, 3>(22, 0) = Mat93::Zero();

			Vec6 W;
			W.block<3, 1>(0, 0) = Vec3(setting_vi_lambda_rot, setting_vi_lambda_rot, setting_vi_lambda_rot);
			W.block<3, 1>(3, 0) = Vec3(setting_vi_lambda_trans, setting_vi_lambda_trans, setting_vi_lambda_trans);

			H += J.transpose() * W.asDiagonal() * J;
			b += -J.transpose() * W.asDiagonal() * r;

			energy = r.transpose() * W.asDiagonal() * r;
		}

		void InertialFrameHessian::setState(Vec15 x_new)
		{
			x = x_new;

			W_v_B_PRE = W_v_B_EvalPT + x.block<3, 1>(6, 0);

			T_WB_PRE = SE3::exp(x.block<6, 1>(0, 0)) * T_WB_EvalPT;
			T_BW_PRE = T_WB_PRE.inverse();

			db_g_PRE = db_g_EvalPT + x.block<3, 1>(9, 0);
			db_a_PRE = db_a_EvalPT + x.block<3, 1>(12, 0);

			std::cout << "InertialFrameHessian STATS: " << fh->frameID << std::endl;
			std::cout << "v: " << std::endl << W_v_B_PRE << std::endl;
			std::cout << "bg: " << std::endl << db_g_PRE << std::endl;
			std::cout << "ba: " << std::endl << db_a_PRE << std::endl;
			std::cout << "T_WB: " << std::endl << T_WB_PRE.matrix() << std::endl;
		}
	}
}

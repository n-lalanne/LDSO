using namespace std;

#include "inertial/InertialFrameHessian.h"
#include "inertial/InertialUtility.h"

namespace ldso {
	namespace inertial {
		void InertialFrameHessian::linearize(shared_ptr<InertialHessian> inertialHessian, double visualWeight, bool force, int nPreIntegrationFactors, int nCombineFactors)
		{
			b.setZero();
			r.setZero();
			energy = 0;

			if (from)
			{
				from->linearize(visualWeight / ((double)nPreIntegrationFactors), force);
			}

			visualWeight /= ((double)nCombineFactors);

			computeResidual(r, inertialHessian->scale_PRE, T_WB_PRE.so3(), T_BW_PRE.so3(), inertialHessian->T_CB.so3(), inertialHessian->T_BC.so3(), fh->PRE_worldToCam.so3(), fh->PRE_camToWorld.so3(), inertialHessian->R_DW_PRE, inertialHessian->R_WD_PRE, T_WB_PRE.translation(), inertialHessian->T_CB.translation(), fh->PRE_camToWorld);

			LOG(INFO) << "r (combine): [" << r.transpose().format(setting_vi_format) << "]";

			if (!setting_vi_fej_window_optimization || force)
			{
				H.setZero();
				J.setZero();
				W.setZero();
				W.block<3, 1>(0, 0) = Vec3(setting_vi_lambda_rot * setting_vi_lambda_rot, setting_vi_lambda_rot* setting_vi_lambda_rot, setting_vi_lambda_rot*setting_vi_lambda_rot);
				W.block<3, 1>(3, 0) = Vec3(setting_vi_lambda_trans*setting_vi_lambda_trans, setting_vi_lambda_trans*setting_vi_lambda_trans, setting_vi_lambda_trans*setting_vi_lambda_trans);

				if (setting_vi_fej_window_optimization)
					computeJacobian(J, inertialHessian->scale_EvalPT, T_WB_EvalPT.so3(), T_BW_EvalPT.so3(), inertialHessian->T_CB.so3(), inertialHessian->T_BC.so3(), fh->worldToCam_evalPT.so3(), fh->worldToCam_evalPT.so3().inverse(), inertialHessian->R_DW_EvalPT, inertialHessian->R_DW_EvalPT.inverse(), T_WB_EvalPT.translation(), inertialHessian->T_CB.translation(), fh->worldToCam_evalPT.inverse());
				else
					computeJacobian(J, inertialHessian->scale_PRE, T_WB_PRE.so3(), T_BW_PRE.so3(), inertialHessian->T_CB.so3(), inertialHessian->T_BC.so3(), fh->PRE_worldToCam.so3(), fh->PRE_camToWorld.so3(), inertialHessian->R_DW_PRE, inertialHessian->R_WD_PRE, T_WB_PRE.translation(), inertialHessian->T_CB.translation(), fh->PRE_camToWorld);

				H += J.transpose() * visualWeight * W.asDiagonal() * J;

				if (from)
				{
					H.block<15, 15>(10, 10) += from->H_from;
				}

				if (to)
				{
					H.block<15, 15>(10, 10) += to->H_to;
				}
			}

			if (from)
			{
				energy += from->energy;
				b.block<15, 1>(10, 0) += from->b_from;

				if (setting_vi_debug)
					LOG(INFO) << "Inertial Pre-Integration (" << fh->frameID << ") dR: [" << (from->preIntegration->delta_R_ij*SO3::exp(from->preIntegration->d_delta_R_ij_dg* db_g_PRE)).log().transpose().format(setting_vi_format) << "]; dv: [" << (from->preIntegration->delta_v_ij + from->preIntegration->d_delta_v_ij_dg * db_g_PRE + from->preIntegration->d_delta_v_ij_da *db_a_PRE).transpose().format(setting_vi_format) << "]; dp: [" << (from->preIntegration->delta_p_ij + from->preIntegration->d_delta_p_ij_dg * db_g_PRE + from->preIntegration->d_delta_p_ij_da *db_a_PRE).transpose().format(setting_vi_format) << "]; dt: " << from->preIntegration->dt_ij << ";";
			}

			if (to)
			{
				b.block<15, 1>(10, 0) += to->b_to;
			}

			b += -J.transpose() * visualWeight * W.asDiagonal() * r;

			energy += r.transpose() * visualWeight * W.asDiagonal() * r;
		}

		void InertialFrameHessian::computeResidual(Vec6 &r, double s, SO3 Rwb, SO3 Rbw, SO3 Rcb, SO3 Rbc, SO3 Rcd, SO3 Rdc, SO3 Rdw, SO3 Rwd, Vec3 pw, Vec3 pc, SE3 Tdc)
		{
			Vec3 dr_p_ds = -exp(s) * (Rwd * (Tdc * pc));

			r.block<3, 1>(0, 0) = (Rwb*Rbc*Rcd*Rdw).log();
			r.block<3, 1>(3, 0) = pw + dr_p_ds;
		}

		void InertialFrameHessian::computeJacobian(Mat625 &J, double s, SO3 Rwb, SO3 Rbw, SO3 Rcb, SO3 Rbc, SO3 Rcd, SO3 Rdc, SO3 Rdw, SO3 Rwd, Vec3 pw, Vec3 pc, SE3 Tdc)
		{
			Vec3 dr_p_ds = -exp(s) * (Rwd * (Tdc * pc));
			Mat33 dr_p_da1 = exp(s) * Rwd.matrix();
			Mat33 dr_p_dq = dr_p_da1 * Rdc.matrix();

			Mat33 JrInv = InertialUtility::JrInv((Rwb*Rbc*Rcd*Rdw).log());

			Mat33 dr_R_dphi = (Rwd * Rdc).matrix();
			Mat33 dr_R_dw = dr_R_dphi * (Rcb * Rbw).matrix();

			//dr_R_dphi
			J.block<3, 3>(0, 3) = JrInv * dr_R_dphi;
			//dr_R_da
			J.block<3, 3>(0, 6) = JrInv * Rwd.matrix();
			//dr_R_dw
			J.block<3, 3>(0, 13) = JrInv * dr_R_dw;


			//dr_p_dq
			J.block<3, 3>(3, 0) = dr_p_dq;
			//dr_p_dphi
			J.block<3, 3>(3, 3) = -dr_p_dq * SO3::hat(pc);
			//dr_p_da
			J.block<3, 3>(3, 6) = -dr_p_da1 * SO3::hat(Tdc * pc);
			//dr_p_ds
			J.block<3, 1>(3, 9) = dr_p_ds;
			//dr_p_du
			J.block<3, 3>(3, 10) = Mat33::Identity();
			//dr_p_dw
			J.block<3, 3>(3, 13) = -SO3::hat(pw);
		}

		void InertialFrameHessian::setState(Vec15 x_new)
		{
			x = x_new;

			W_v_B_PRE = W_v_B_EvalPT + x.block<3, 1>(6, 0);

			T_WB_PRE = SE3::exp(x.block<6, 1>(0, 0)) * T_WB_EvalPT;
			T_BW_PRE = T_WB_PRE.inverse();

			db_g_PRE = db_g_EvalPT + x.block<3, 1>(9, 0);
			db_a_PRE = db_a_EvalPT + x.block<3, 1>(12, 0);

			if (setting_vi_debug)
				LOG(INFO) << "Inertial Frame Hessian (" << fh->frameID << ") u: [" << T_WB_PRE.log().transpose().segment<3>(0).format(setting_vi_format) << "]; omega: [" << T_WB_PRE.log().transpose().segment<3>(3).format(setting_vi_format) << "]; v: [" << W_v_B_PRE.transpose().format(setting_vi_format) << "]; bg: [" << db_g_PRE.transpose().format(setting_vi_format) << "]; ba: [" << db_a_PRE.transpose().format(setting_vi_format) << "];";
		}
	}
}

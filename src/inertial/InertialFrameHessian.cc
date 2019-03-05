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

			if (!setting_vi_fej_window_optimization || force)
			{
				H.setZero();
				J.setZero();
				W.setZero();
				W.block<3, 1>(0, 0) = setting_vi_lambda_rot * setting_vi_lambda_rot * Vec3::Ones();
				W.block<3, 1>(3, 0) = setting_vi_lambda_trans * setting_vi_lambda_trans * Vec3::Ones();

				if (setting_vi_fej_window_optimization)
					computeJacobian(J, inertialHessian->scale_PRE, T_WB_EvalPT.so3(), T_WB_EvalPT.so3().inverse(), inertialHessian->T_CB.so3(), inertialHessian->T_BC.so3(), fh->worldToCam_evalPT.so3(), fh->worldToCam_evalPT.so3().inverse(), inertialHessian->R_DW_PRE, inertialHessian->R_WD_PRE, T_WB_EvalPT.translation(), inertialHessian->T_CB.translation(), fh->worldToCam_evalPT.inverse());
				else
					computeJacobian(J, inertialHessian->scale_PRE, T_WB_PRE.so3(), T_BW_PRE.so3(), inertialHessian->T_CB.so3(), inertialHessian->T_BC.so3(), fh->PRE_worldToCam.so3(), fh->PRE_camToWorld.so3(), inertialHessian->R_DW_PRE, inertialHessian->R_WD_PRE, T_WB_PRE.translation(), inertialHessian->T_CB.translation(), fh->PRE_camToWorld);

				H.block<16, 16>(0, 0).triangularView<Eigen::Upper>() = J.block<6, 16>(0, 0).transpose() * visualWeight * W.asDiagonal() * J.block<6, 16>(0, 0);

				lastVisualWeight = visualWeight;

				if (from)
				{
					H.block<15, 15>(10, 10).triangularView<Eigen::Upper>() += from->H_from;
				}

				if (to)
				{
					H.block<15, 15>(10, 10).triangularView<Eigen::Upper>() += to->H_to;
				}
			}

			if (from)
			{
				energy += from->energy;
				b.block<15, 1>(10, 0) += from->b_from;

				if (setting_vi_debug)
					//LOG(INFO) << "r (pre): [" << (visualWeight * ((double)nCombineFactors) / ((double)nPreIntegrationFactors) * from->W.selfadjointView<Eigen::Upper>().toDenseMatrix() * from->r).transpose().format(setting_vi_format) << "] - Energy: " << (from->r.transpose() * visualWeight * ((double)nCombineFactors) / ((double)nPreIntegrationFactors) * from->W.selfadjointView<Eigen::Upper>().toDenseMatrix() * from->r);
					LOG(INFO) << "r (pre): [" << (from->r).transpose().format(setting_vi_format) << "] - Energy: " << (from->r.transpose() * visualWeight * ((double)nCombineFactors) / ((double)nPreIntegrationFactors) * from->W.selfadjointView<Eigen::Upper>().toDenseMatrix() * from->r);

				if (setting_vi_debug)
					LOG(INFO) << "Gravity: [" << (T_WB_EvalPT.so3() * from->preIntegration->g_mean).format(setting_vi_format) << "] - error: " << (T_WB_EvalPT.so3() * from->preIntegration->g_mean - Vec3(0, 0, 9.81)).format(setting_vi_format) << " (" << (T_WB_EvalPT.so3() * from->preIntegration->g_mean - Vec3(0, 0, 9.81)).norm() << ")";

				if (setting_vi_debug)
					LOG(INFO) << "Inertial Pre-Integration (" << fh->frameID << ") dR: [" << (from->preIntegration->delta_R_ij*SO3::exp(from->preIntegration->d_delta_R_ij_dg* db_g_PRE)).log().transpose().format(setting_vi_format) << "]; dv: [" << (T_WB_PRE.so3() * (from->preIntegration->delta_v_ij + from->preIntegration->d_delta_v_ij_dg * db_g_PRE + from->preIntegration->d_delta_v_ij_da *db_a_PRE) - Vec3(0, 0, 9.81*from->preIntegration->dt_ij)).transpose().format(setting_vi_format) << "]; dp: [" << (T_WB_PRE.so3() *(from->preIntegration->delta_p_ij + from->preIntegration->d_delta_p_ij_dg * db_g_PRE + from->preIntegration->d_delta_p_ij_da *db_a_PRE)).transpose().format(setting_vi_format) << "]; dt: " << from->preIntegration->dt_ij << ";";
			}

			if (to)
			{
				b.block<15, 1>(10, 0) += to->b_to;
			}

			if (setting_vi_fej_window_optimization)
			{
				H.block<16, 16>(0, 0).triangularView<Eigen::Upper>() *= visualWeight / lastVisualWeight;
			}

			//LOG(INFO) << "eigen H: " << H.eigenvalues().format(setting_vi_format);

			if (setting_vi_debug)
				//LOG(INFO) << "r (combine): [" << (visualWeight * W.asDiagonal() * r).transpose().format(setting_vi_format) << "] - Energy: " << (r.transpose() * visualWeight * W.asDiagonal() * r) << " - Weight: " << (visualWeight * W).transpose().format(setting_vi_format);
				LOG(INFO) << "r (combine): [" << r.transpose().format(setting_vi_format) << "] - Energy: " << (r.transpose() * visualWeight * W.asDiagonal() * r);

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

		void InertialFrameHessian::setCurrentStateAsEvalPt()
		{
			T_WB_EvalPT = T_WB_PRE;
			W_v_B_EvalPT = W_v_B_PRE;
			db_g_EvalPT = db_g_PRE;
			db_a_EvalPT = db_a_PRE;
			x.setZero();
			x_backup.setZero();
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
				LOG(INFO) << "Inertial Frame Hessian (" << fh->frameID << ") u: [" << T_WB_PRE.log().transpose().segment<3>(0).format(setting_vi_format) << "]; omega: [" << T_WB_PRE.log().transpose().segment<3>(3).format(setting_vi_format) << "]; v: [" << W_v_B_PRE.transpose().format(setting_vi_format) << "]; bg: [" << (db_g_PRE + b_g_lin).transpose().format(setting_vi_format) << "]; ba: [" << (db_a_PRE + b_a_lin).transpose().format(setting_vi_format) << "];";
		}
	}
}

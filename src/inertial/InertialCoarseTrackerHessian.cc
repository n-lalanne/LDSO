using namespace std;

#include "inertial/InertialCoarseTrackerHessian.h"
#include "inertial/InertialFrameFrameHessian.h"
#include "inertial/InertialFrameHessian.h"
#include "util/MatrixInverter.h"

namespace ldso {
	namespace inertial {

		InertialCoarseTrackerHessian::InertialCoarseTrackerHessian()
		{
			Vec25 s;
			s << SCALE_XI_TRANS, SCALE_XI_TRANS, SCALE_XI_TRANS, SCALE_XI_ROT, SCALE_XI_ROT, SCALE_XI_ROT, SCALE_VI_ROT, SCALE_VI_ROT, SCALE_VI_ROT, SCALE_VI_S, SCALE_VI_TRANS, SCALE_VI_TRANS, SCALE_VI_TRANS, SCALE_VI_ROT, SCALE_VI_ROT, SCALE_VI_ROT, SCALE_VI_V, SCALE_VI_V, SCALE_VI_V, SCALE_VI_B, SCALE_VI_B, SCALE_VI_B, SCALE_VI_B, SCALE_VI_B, SCALE_VI_B;
			S = s.asDiagonal().toDenseMatrix();
		}

		void InertialCoarseTrackerHessian::marginalize()
		{
			if (setting_vi_enable && setting_vi_enable_coarse_tracker) {
				fix_i = false;

				Mat1515 J_i = Mat1515::Zero();
				Mat1515 J_j = Mat1515::Zero();
				Vec15 r_pr = Vec15::Zero();

				InertialFrameFrameHessian::computeResidual(r_pr, preIntegration, Tw_i.translation(), Tw_j.translation(), Tw_i.so3().inverse(), Tw_j.so3().inverse(), Tw_j.so3(), v_i, v_j, bg_i, bg_j, ba_i, ba_j, lin_bias_g, lin_bias_g, lin_bias_a, lin_bias_a);
				InertialFrameFrameHessian::computeJacobian(J_i, J_j, preIntegration, Tw_i.translation(), Tw_j.translation(), Tw_i.so3().inverse(), Tw_j.so3().inverse(), Tw_j.so3(), v_i, v_j, bg_i, bg_j, ba_i, ba_j, lin_bias_g, lin_bias_g + bg_j, lin_bias_a, lin_bias_a + ba_j);

				J_i = J_i * S.block<15, 15>(10, 10);
				J_j = J_j * S.block<15, 15>(10, 10);

				Mat1515 Hbb = Mat1515::Zero();
				Mat1515 Hba = Mat1515::Zero(15, 15);
				Vec15 bb = Vec15::Zero();

				Hbb.triangularView<Eigen::Upper>() = setting_vi_marginalization_weight * J_i.transpose() * W.selfadjointView<Eigen::Upper>() * J_i + HM_I.selfadjointView<Eigen::Upper>().toDenseMatrix();
				Hba = setting_vi_marginalization_weight * J_i.transpose() * W.selfadjointView<Eigen::Upper>() * J_j;

				bb = -setting_vi_marginalization_weight * J_i.transpose() * W.selfadjointView<Eigen::Upper>() * r_pr + bM_I;

				Mat1515 HabHbbinv;
				HabHbbinv = Hba.transpose() * util::MatrixInverter::invertPosDef(Hbb, setting_use_fast_matrix_inverter).selfadjointView<Eigen::Upper>();

				HM_I.triangularView<Eigen::Upper>() = setting_vi_marginalization_weight * J_j.transpose() * W.selfadjointView<Eigen::Upper>() * J_j - HabHbbinv * Hba;
				bM_I = -setting_vi_marginalization_weight * J_j.transpose() * W.selfadjointView<Eigen::Upper>() * r_pr - HabHbbinv * bb;

				Tw_i = Tw_j;
				v_i = v_j;
				bg_i = bg_j;
				ba_i = ba_j;
			}
		}

		void InertialCoarseTrackerHessian::restore() {
			if (setting_vi_enable && setting_vi_enable_coarse_tracker) {
				applyStep(-step);
			}
		}

		void InertialCoarseTrackerHessian::compute(double visualWeight, SE3 T_id, SE3 T_ji, double lambda)
		{
			Mat1515 J_i = Mat1515::Zero();
			Mat1515 J_j = Mat1515::Zero();
			Vec15 r_pr = Vec15::Zero();
			Vec6 r_co = Vec6::Zero();
			Mat625 J_co = Mat625::Zero();

			energy = 0;
			H_I.setZero();
			H_I_sc.setZero();
			b_I_sc.setZero();

			if (setting_vi_enable && setting_vi_enable_coarse_tracker) {
				InertialFrameFrameHessian::computeResidual(r_pr, preIntegration, Tw_i.translation(), Tw_j.translation(), Tw_i.so3().inverse(), Tw_j.so3().inverse(), Tw_j.so3(), v_i, v_j, bg_i, bg_j, ba_i, ba_j, lin_bias_g, lin_bias_g, lin_bias_a, lin_bias_a);
				InertialFrameFrameHessian::computeJacobian(J_i, J_j, preIntegration, Tw_i.translation(), Tw_j.translation(), Tw_i.so3().inverse(), Tw_j.so3().inverse(), Tw_j.so3(), v_i, v_j, bg_i, bg_j, ba_i, ba_j, lin_bias_g, lin_bias_g, lin_bias_a, lin_bias_a);

				//LOG(INFO) << "Bias Lin: (g): " << lin_bias_g.transpose().format(setting_vi_format) << "(a): " << lin_bias_a.transpose().format(setting_vi_format);

				SE3 Tcd = T_ji * T_id;

				InertialFrameHessian::computeResidual(r_co, scale, Tw_j.so3(), Tw_j.so3().inverse(), T_bc.so3().inverse(), T_bc.so3(), Tcd.so3(), Tcd.so3().inverse(), R_wd.inverse(), R_wd, Tw_j.translation(), T_bc.inverse().translation(), Tcd.inverse());
				InertialFrameHessian::computeJacobian(J_co, scale, Tw_j.so3(), Tw_j.so3().inverse(), T_bc.so3().inverse(), T_bc.so3(), Tcd.so3(), Tcd.so3().inverse(), R_wd.inverse(), R_wd, Tw_j.translation(), T_bc.inverse().translation(), Tcd.inverse());

				//LOG(INFO) << "r (pre integration): " << r_pr.format(setting_vi_format) << "; r (join): " <<  r_co.format(setting_vi_format);

				J_co = J_co * S;

				W.setZero();
				W.block<9, 9>(0, 0).triangularView<Eigen::Upper>() = preIntegration->Sigma_ij;
				W.block<6, 6>(9, 9).triangularView<Eigen::Upper>() = preIntegration->Sigma_bd * preIntegration->dt_ij;
				W = setting_vi_lambda_coarse_tracker * util::MatrixInverter::invertPosDef(W, setting_use_fast_matrix_inverter);

				w.setZero();
				w.block<3, 1>(0, 0) = setting_vi_lambda_coarse_tracker * setting_vi_lambda_rot * setting_vi_lambda_rot * Vec3::Ones();
				w.block<3, 1>(3, 0) = setting_vi_lambda_coarse_tracker * setting_vi_lambda_trans * Vec3::Ones();


				if (fix_i)
				{
					J_j = J_j * S.block<15, 15>(10, 10);
					Hbb_inv = MatXX::Zero(15, 15);
					bb = VecX::Zero(15);
					Hab = MatXX::Zero(8, 15);

					Mat1515 Hj;

					Hj.triangularView<Eigen::Upper>() = J_j.transpose() * visualWeight * W.selfadjointView<Eigen::Upper>() * J_j;
					Hj.block<6, 6>(0, 0).triangularView<Eigen::Upper>() += J_co.block<6, 6>(0, 10).transpose() * visualWeight * w.asDiagonal() * J_co.block<6, 6>(0, 10);

					for (int i = 0; i < 15; i++)
						Hj(i, i) *= (1 + lambda);

					Hbb_inv = util::MatrixInverter::invertPosDef(Hj, setting_use_fast_matrix_inverter);

					bb += -J_j.transpose() * visualWeight * W.selfadjointView<Eigen::Upper>() * r_pr;
					bb += -J_co.block<6, 15>(0, 10).transpose() *  visualWeight * w.asDiagonal() * r_co;
					Hab.block<6, 15>(0, 0) = J_co.block<6, 6>(0, 0).transpose() * visualWeight * w.asDiagonal() * J_co.block<6, 15>(0, 10);
				}
				else
				{
					J_i = J_i * S.block<15, 15>(10, 10);
					J_j = J_j * S.block<15, 15>(10, 10);
					Hbb_inv = MatXX::Zero(30, 30);

					bb = VecX::Zero(30);
					Hab = MatXX::Zero(8, 30);

					MatXX Hbb = MatXX::Zero(30, 30);

					Hbb.block<15, 15>(0, 0).triangularView<Eigen::Upper>() = J_i.transpose() * visualWeight * W.selfadjointView<Eigen::Upper>() * J_i + visualWeight * HM_I.selfadjointView<Eigen::Upper>().toDenseMatrix();
					Hbb.block<15, 15>(0, 15) = J_i.transpose() * visualWeight * W.selfadjointView<Eigen::Upper>() * J_j;
					//Hbb.block<15, 15>(15, 0) = Hbb.block<15, 15>(0, 15).transpose();
					Hbb.block<15, 15>(15, 15).triangularView<Eigen::Upper>() = J_j.transpose() * visualWeight * W.selfadjointView<Eigen::Upper>() * J_j;
					Hbb.block<6, 6>(15, 15).triangularView<Eigen::Upper>() += J_co.block<6, 6>(0, 10).transpose() * visualWeight * w.asDiagonal() * J_co.block<6, 6>(0, 10);

					for (int i = 0; i < 30; i++) {
						Hbb(i, i) *= (1 + lambda);
					}

					Hbb_inv = util::MatrixInverter::invertPosDef(Hbb, setting_use_fast_matrix_inverter);

					bb.block<15, 1>(0, 0) += -J_i.transpose() * visualWeight * W.selfadjointView<Eigen::Upper>() * r_pr + visualWeight * (bM_I - HM_I.selfadjointView<Eigen::Upper>() * S.block<15, 15>(10, 10).inverse() * (x_i - x_backup_i));
					bb.block<15, 1>(15, 0) += -J_j.transpose() * visualWeight * W.selfadjointView<Eigen::Upper>() * r_pr;
					bb.block<15, 1>(15, 0) += -J_co.block<6, 15>(0, 10).transpose() *  visualWeight * w.asDiagonal() * r_co;
					Hab.block<6, 15>(0, 15) = J_co.block<6, 6>(0, 0).transpose() * visualWeight * w.asDiagonal() *  J_co.block<6, 15>(0, 10);
				}

				H_I.block<6, 6>(0, 0).triangularView<Eigen::Upper>() = J_co.block<6, 6>(0, 0).transpose() * visualWeight * w.asDiagonal() *  J_co.block<6, 6>(0, 0);
				b_I.block<6, 1>(0, 0) = -J_co.block<6, 6>(0, 0).transpose() * visualWeight * w.asDiagonal() * r_co;

				MatXX HabHbbinv;
				HabHbbinv = Hab * Hbb_inv.selfadjointView<Eigen::Upper>();

				H_I_sc.triangularView<Eigen::Upper>() = HabHbbinv * Hab.transpose();
				b_I_sc += HabHbbinv * bb;

				energy += r_pr.transpose() * visualWeight * W.selfadjointView<Eigen::Upper>() * r_pr;
				energy += r_co.transpose() * visualWeight * w.asDiagonal() * r_co;
			}
		}

		void InertialCoarseTrackerHessian::setValues(std::vector <shared_ptr<internal::FrameHessian>> &frameHessians, shared_ptr<inertial::InertialHessian> Hinertial)
		{
			T_bc = Hinertial->T_BC;
			R_wd = Hinertial->R_WD_PRE;

			if (setting_vi_enable) {
				assert(frameHessians.size() > 0);
				shared_ptr<internal::FrameHessian> fh = frameHessians.back();

				v_i = fh->inertialFrameHessian->W_v_B_PRE;
				v_j = v_i;

				Tw_i = fh->inertialFrameHessian->T_WB_PRE;
				Tw_j = Tw_i;

				bg_i = Vec3::Zero();
				bg_j = Vec3::Zero();

				ba_i = Vec3::Zero();
				ba_j = Vec3::Zero();

				scale = Hinertial->scale_PRE;
				
				lin_bias_g = fh->inertialFrameHessian->b_g_lin + fh->inertialFrameHessian->db_g_PRE;
				lin_bias_a = fh->inertialFrameHessian->b_a_lin + fh->inertialFrameHessian->db_a_PRE;

				fix_i = true;
				HM_I.setZero();
				bM_I.setZero();
			}
		}

		void InertialCoarseTrackerHessian::applyStep(VecX s) {
			if (setting_vi_enable && setting_vi_enable_coarse_tracker)
			{
				if (!fix_i)
				{
					Tw_i = SE3::exp(s.block<6, 1>(0, 0)) * Tw_i;
					v_i += s.block<3, 1>(6, 0);
					bg_i += s.block<3, 1>(9, 0);
					ba_i += s.block<3, 1>(12, 0);
					Tw_j = SE3::exp(s.block<6, 1>(0 + 15, 0)) * Tw_j;
					v_j += s.block<3, 1>(6 + 15, 0);
					bg_j += s.block<3, 1>(9 + 15, 0);
					ba_j += s.block<3, 1>(12 + 15, 0);
					x_i += s.block<15, 1>(0, 0);
					x_j += s.block<15, 1>(15, 0);
				}
				else
				{
					Tw_j = SE3::exp(s.block<6, 1>(0, 0)) * Tw_j;
					v_j += s.block<3, 1>(6, 0);
					bg_j += s.block<3, 1>(9, 0);
					ba_j += s.block<3, 1>(12, 0);
					x_j += s.block<15, 1>(0, 0);
					//LOG(INFO) << "step: " << s.transpose().format(setting_vi_format);
				}
			}
		}

		void InertialCoarseTrackerHessian::update(Vec8 x) {
			if (setting_vi_enable && setting_vi_enable_coarse_tracker) {
				step = Hbb_inv.selfadjointView<Eigen::Upper>() * (bb - Hab.transpose() * x);

				if (!fix_i)
				{
					step.block<15, 1>(0, 0) = S.block<15, 15>(10, 10) * step.block<15, 1>(0, 0);
					step.block<15, 1>(15, 0) = S.block<15, 15>(10, 10) * step.block<15, 1>(15, 0);
				}
				else
				{
					step.block<15, 1>(0, 0) = S.block<15, 15>(10, 10) * step.block<15, 1>(0, 0);
				}
				applyStep(step);
			}
		}
		void InertialCoarseTrackerHessian::backup() {
			if (setting_vi_enable && setting_vi_enable_coarse_tracker) {
				x_backup_i = Vec15::Zero();
				x_backup_j = Vec15::Zero();

				x_backup_i.block<6, 1>(0, 0) = Tw_i.log();
				x_backup_j.block<6, 1>(0, 0) = Tw_j.log();

				x_backup_i.block<3, 1>(6, 0) = v_i;
				x_backup_j.block<3, 1>(6, 0) = v_j;

				x_backup_i.block<3, 1>(9, 0) = bg_i;
				x_backup_j.block<3, 1>(9, 0) = bg_j;

				x_backup_i.block<3, 1>(12, 0) = ba_i;
				x_backup_j.block<3, 1>(12, 0) = ba_j;

				x_i = x_backup_i;
				x_j = x_backup_j;

				if (setting_vi_debug)
					LOG(INFO) << "xi: " << x_i.transpose().format(setting_vi_format) << "xi_backup: " << x_backup_i.transpose().format(setting_vi_format);
			}
		}

		void InertialCoarseTrackerHessian::reset() {
			if (setting_vi_enable && setting_vi_enable_coarse_tracker) {
				Tw_i = SE3::exp(x_backup_i.block<6, 1>(0, 0));
				Tw_j = SE3::exp(x_backup_j.block<6, 1>(0, 0));

				v_i = x_backup_i.block<3, 1>(6, 0);
				v_j = x_backup_j.block<3, 1>(6, 0);

				bg_i = x_backup_i.block<3, 1>(9, 0);
				bg_j = x_backup_j.block<3, 1>(9, 0);

				ba_i = x_backup_i.block<3, 1>(12, 0);
				ba_j = x_backup_j.block<3, 1>(12, 0);

				x_i = x_backup_i;
				x_j = x_backup_j;
			}
		}
	}
}
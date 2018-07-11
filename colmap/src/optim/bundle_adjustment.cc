// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "optim/bundle_adjustment.h"

#include <iomanip>
#include <mpi.h>
#include <float.h>
#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

#include "base/cost_functions.h"
#include "base/projection.h"
#include "util/misc.h"
#include "util/threading.h"
#include "util/timer.h"

using std::cout;
using std::endl;

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// BundleAdjustmentConfig
////////////////////////////////////////////////////////////////////////////////

BundleAdjustmentConfig::BundleAdjustmentConfig() {}

size_t BundleAdjustmentConfig::NumImages() const { return image_ids_.size(); }

size_t BundleAdjustmentConfig::NumPoints() const {
  return variable_point3D_ids_.size() + constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantCameras() const {
  return constant_camera_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantPoses() const {
  return constant_poses_.size();
}

size_t BundleAdjustmentConfig::NumConstantTvecs() const {
  return constant_tvecs_.size();
}

size_t BundleAdjustmentConfig::NumVariablePoints() const {
  return variable_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumConstantPoints() const {
  return constant_point3D_ids_.size();
}

size_t BundleAdjustmentConfig::NumResiduals(
    const Reconstruction& reconstruction) const {
  // Count the number of observations for all added images.
  size_t num_observations = 0;
  for (const image_t image_id : image_ids_) {
    num_observations += reconstruction.Image(image_id).NumPoints3D();
  }

  // Count the number of observations for all added 3D points that are not
  // already added as part of the images above.

  auto NumObservationsForPoint = [this,
                                  &reconstruction](const point3D_t point3D_id) {
    size_t num_observations_for_point = 0;
    const auto& point3D = reconstruction.Point3D(point3D_id);
    for (const auto& track_el : point3D.Track().Elements()) {
      if (image_ids_.count(track_el.image_id) == 0) {
        num_observations_for_point += 1;
      }
    }
    return num_observations_for_point;
  };

  for (const auto point3D_id : variable_point3D_ids_) {
    num_observations += NumObservationsForPoint(point3D_id);
  }
  for (const auto point3D_id : constant_point3D_ids_) {
    num_observations += NumObservationsForPoint(point3D_id);
  }

  return 2 * num_observations;
}

void BundleAdjustmentConfig::AddImage(const image_t image_id) {
  image_ids_.insert(image_id);
}

bool BundleAdjustmentConfig::HasImage(const image_t image_id) const {
  return image_ids_.find(image_id) != image_ids_.end();
}

void BundleAdjustmentConfig::RemoveImage(const image_t image_id) {
  image_ids_.erase(image_id);
}

void BundleAdjustmentConfig::SetConstantCamera(const camera_t camera_id) {
  constant_camera_ids_.insert(camera_id);
}

void BundleAdjustmentConfig::SetVariableCamera(const camera_t camera_id) {
  constant_camera_ids_.erase(camera_id);
}

bool BundleAdjustmentConfig::IsConstantCamera(const camera_t camera_id) const {
  return constant_camera_ids_.find(camera_id) != constant_camera_ids_.end();
}

void BundleAdjustmentConfig::SetConstantPose(const image_t image_id) {
  CHECK(HasImage(image_id));
  CHECK(!HasConstantTvec(image_id));
  constant_poses_.insert(image_id);
}

void BundleAdjustmentConfig::SetVariablePose(const image_t image_id) {
  constant_poses_.erase(image_id);
}

bool BundleAdjustmentConfig::HasConstantPose(const image_t image_id) const {
  return constant_poses_.find(image_id) != constant_poses_.end();
}

void BundleAdjustmentConfig::SetConstantTvec(const image_t image_id,
                                             const std::vector<int>& idxs) {
  CHECK_GT(idxs.size(), 0);
  CHECK_LE(idxs.size(), 3);
  CHECK(HasImage(image_id));
  CHECK(!HasConstantPose(image_id));
  CHECK(!VectorContainsDuplicateValues(idxs))
      << "Tvec indices must not contain duplicates";
  constant_tvecs_.emplace(image_id, idxs);
}

void BundleAdjustmentConfig::RemoveConstantTvec(const image_t image_id) {
  constant_tvecs_.erase(image_id);
}

bool BundleAdjustmentConfig::HasConstantTvec(const image_t image_id) const {
  return constant_tvecs_.find(image_id) != constant_tvecs_.end();
}

const std::unordered_set<image_t>& BundleAdjustmentConfig::Images() const {
  return image_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::VariablePoints()
    const {
  return variable_point3D_ids_;
}

const std::unordered_set<point3D_t>& BundleAdjustmentConfig::ConstantPoints()
    const {
  return constant_point3D_ids_;
}

const std::vector<int>& BundleAdjustmentConfig::ConstantTvec(
    const image_t image_id) const {
  return constant_tvecs_.at(image_id);
}

void BundleAdjustmentConfig::AddVariablePoint(const point3D_t point3D_id) {
  CHECK(!HasConstantPoint(point3D_id));
  variable_point3D_ids_.insert(point3D_id);
}

void BundleAdjustmentConfig::AddConstantPoint(const point3D_t point3D_id) {
  CHECK(!HasVariablePoint(point3D_id));
  constant_point3D_ids_.insert(point3D_id);
}

bool BundleAdjustmentConfig::HasPoint(const point3D_t point3D_id) const {
  return HasVariablePoint(point3D_id) || HasConstantPoint(point3D_id);
}

bool BundleAdjustmentConfig::HasVariablePoint(
    const point3D_t point3D_id) const {
  return variable_point3D_ids_.find(point3D_id) != variable_point3D_ids_.end();
}

bool BundleAdjustmentConfig::HasConstantPoint(
    const point3D_t point3D_id) const {
  return constant_point3D_ids_.find(point3D_id) != constant_point3D_ids_.end();
}

void BundleAdjustmentConfig::RemoveVariablePoint(const point3D_t point3D_id) {
  variable_point3D_ids_.erase(point3D_id);
}

void BundleAdjustmentConfig::RemoveConstantPoint(const point3D_t point3D_id) {
  constant_point3D_ids_.erase(point3D_id);
}

////////////////////////////////////////////////////////////////////////////////
// BundleAdjuster
////////////////////////////////////////////////////////////////////////////////

ceres::LossFunction* BundleAdjuster::Options::CreateLossFunction() const {
  ceres::LossFunction* loss_function = nullptr;
  switch (loss_function_type) {
    case LossFunctionType::TRIVIAL:
      loss_function = new ceres::TrivialLoss();
      break;
    case LossFunctionType::CAUCHY:
      loss_function = new ceres::CauchyLoss(loss_function_scale);
      break;
  }
  return loss_function;
}

bool BundleAdjuster::Options::Check() const {
  CHECK_OPTION_GE(loss_function_scale, 0);
  return true;
}

BundleAdjuster::BundleAdjuster(const Options& options,
                               const BundleAdjustmentConfig& config)
    : options_(options), config_(config) {
  CHECK(options_.Check());
}

const std::unordered_set<camera_t>& BundleAdjuster::CameraIds() {
  return camera_ids_;
}

const std::unordered_set<image_t>& BundleAdjuster::Images() {
  return config_.Images();
}

Eigen::Vector3d QuaternionToAngleAxis(const Eigen::Vector4d& qvec) {
  double angle_axis_d[3];
  double qvec_d[4];
  for (int i = 0; i < 4; i++) {
    qvec_d[i] = qvec[i];
  }
  ceres::QuaternionToAngleAxis(qvec_d, angle_axis_d);
  
  Eigen::Vector3d angle_axis;
  for (int i = 0; i < 3; i++) {
    angle_axis[i] = angle_axis_d[i];
  }
  return angle_axis;
}

bool BundleAdjuster::CollectResults(int rank, Reconstruction* recon) {

  if (rank == 0) {
    
  } else {
    
  }
  return true;
}

bool BundleAdjuster::InitDBAParameters(Reconstruction *reconstruction) {
  size_t num_observations = reconstruction->ComputeNumObservations();
  ro_intri_[0] = sqrt(num_observations) * options_.ro_focal;
  ro_intri_[1] = sqrt(num_observations) * options_.ro_principal;
  ro_intri_[2] = ro_intri_[1];
  ro_intri_[3] = sqrt(num_observations) * options_.ro_distortion;
  
  ro_extri_[0] = sqrt(num_observations / reconstruction->NumRegImages()) * options_.ro_rotation;
  ro_extri_[1] = sqrt(num_observations / reconstruction->NumRegImages()) * options_.ro_center;
  
  ro_point_[0] = sqrt(num_observations / reconstruction->NumPoints3D()) * options_.ro_point;
  ro_point_[1] = ro_point_[0];
  ro_point_[2] = ro_point_[0];
  return true;
}

// Implemented according to "Distributed Very Large Scale Bundle Adjustment by Global Camera Consensus, Runze Zhang et al, ICCV 2017"
bool BundleAdjuster::DistributedSolve(Reconstruction* reconstruction) {
  
  int rank;
  int nprocs;
  
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  
  for (const image_t image_id : config_.Images()) {
    class Image &image = reconstruction->Image(image_id);
    image.qvec_error_ = Eigen::Vector4d::Constant(0);
    image.angle_error_ = Eigen::Vector3d::Constant(0);
    image.tvec_error_ = Eigen::Vector3d::Constant(0);
  }
  
  EIGEN_STL_UMAP(image_t, Extrinsic) prev_images_avg;
  EIGEN_STL_UMAP(camera_t, Intrinsic) prev_cameras_avg;

  InitDBAParameters(reconstruction);
  double pow_alpha = 1;
  
  for (int iter = 0; iter < options_.dba_max_iterations; iter++) {

    cout << "ro intri " << ro_intri_[0] << " " << ro_intri_[3] << endl;
    cout << "ro extri " << ro_extri_[0] << " " << ro_extri_[1] << endl;
    cout << "ro point " << ro_point_[0] << endl;
   
    double primal_residual = 0;
    double dual_residual = 0;
    double dual_points_residual = 0;
    
    reconstruction->PrintSummary();
    Solve(reconstruction);
    
    if (iter > 0) {
      for (auto &point : reconstruction->points3D_) {
        
        dual_points_residual += ro_point_[0] * ro_point_[0] * (point.second.XYZ() - reconstruction->prev_points3D_[point.first].XYZ()).norm();
      }
      MPI_Reduce(&dual_points_residual, &dual_residual, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      if (rank == 0) {
        printf("points dual residual: %lf\n", dual_residual);
      }
    }
    
    int extri_count = 0;
    int intri_count = 0;
    
    if (rank == 0) {
      
      EIGEN_STL_UMAP(image_t, int) images_count;
      EIGEN_STL_UMAP(image_t, Extrinsic) images_avg;
      EIGEN_STL_UMAP(image_t, vector<Extrinsic>) images_all;
      
      EIGEN_STL_UMAP(camera_t, int) cameras_count;
      EIGEN_STL_UMAP(camera_t, Intrinsic) cameras_avg;
      EIGEN_STL_UMAP(camera_t, vector<Intrinsic>) cameras_all;
      
      for (const camera_t camera_id : camera_ids_) {
        Camera& camera = reconstruction->Camera(camera_id);
        cameras_count[camera_id] = 1;
        Intrinsic intri;
        intri.camera_id = camera_id;
        for (int i = 0; i < 4; i++) {
          intri.params[i] = camera.Params()[i];
        }
        cameras_avg[camera_id] = intri;
      }
      
      for (const image_t image_id : config_.Images()) {
        class Image &image = reconstruction->Image(image_id);
        images_count[image_id] = 1;
        Extrinsic extri;
        extri.image_id = image_id;
        extri.qvec = image.Qvec();
        extri.tvec = image.Tvec();
        images_avg[image_id] = extri;
      }
      
      for (int i = 1; i < nprocs; i++) {
        MPI_Status status;
        int count;
        
        // collect intrinsics
        MPI_Recv(p_intri_, sizeof(Intrinsic) * 20, MPI_BYTE, i, i, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_BYTE, &count);
        printf("root received %d bytes intrinsics from process %d\n", count, i);

        count /= sizeof(Intrinsic);
        for (int j = 0; j < count; j++) {
          camera_t id = p_intri_[j].camera_id;
          cout << "received intrinsic from process " << i << " focal length " <<p_intri_[j].params[0] << endl;
          if (cameras_count.find(id) == cameras_count.end()) {
            cameras_count[id] = 1;
            cameras_avg[id] = p_intri_[j];
          } else {
            cameras_count[id]++;
            cameras_avg[id].params += p_intri_[j].params;
          }
          cameras_all[id].push_back(p_intri_[j]);
        }
        
        // collect extrinsics
        MPI_Recv(p_extri_, sizeof(Extrinsic) * 20000, MPI_BYTE, i, i, MPI_COMM_WORLD, &status);
        
        MPI_Get_count(&status, MPI_BYTE, &count);
        printf("root received %d bytes data from process %d\n", count, i);
        count /= sizeof(Extrinsic);
        for (int j = 0; j < count; j++) {
          image_t image_id = p_extri_[j].image_id;
          if (images_count.find(image_id) == images_count.end()) {
            images_count[image_id] = 1;
            images_avg[image_id] = p_extri_[j];
          } else {
            images_count[image_id]++;
            Extrinsic &avg = images_avg[image_id];
            Eigen::Vector3d old_center = ProjectionCenterFromParameters(avg.qvec, avg.tvec);
            Eigen::Vector3d new_center = ProjectionCenterFromParameters(p_extri_[j].qvec, p_extri_[j].tvec);
            Eigen::Vector3d avg_center = (old_center * (images_count[image_id] - 1.0) + new_center) / images_count[image_id];
            //cout << "projection center before interpolation " << old_center << endl;
            //cout << "avg tvec " << avg.tvec << endl;
            //cout << "projection center of new " << new_center << endl;
            //cout << "new tvec " << p_extri[j].tvec << endl;
            
            avg.qvec = AverageQuaternions({avg.qvec, p_extri_[j].qvec}, {double(images_count[image_id] - 1), 1.0});
            //avg.tvec = (avg.tvec * (images_count[image_id] - 1.0) + p_extri[j].tvec) / double(images_count[image_id]);
            avg.tvec = -QuaternionToRotationMatrix(avg.qvec) * avg_center;
          }
          
          images_all[image_id].push_back(p_extri_[j]);
          //cout << "image_id: " << p_extri[j].image_id << p_extri[j].qvec << " " << p_extri[j].tvec << endl;
        }
      }
      
      // get average of intrinsics
      
      for (auto &avg : cameras_avg) {
        camera_t id = avg.first;
        cameras_avg[id].params /= cameras_count[id];
        p_intri_[intri_count++] = cameras_avg[id];
        cout << "root get intrinsics average " << " focal length " << avg.second.params[0] << " principal x " << avg.second.params[1] << " principal y " << avg.second.params[2] << endl;
        
        // accumulate primal residual
        if (cameras_count[id] > 1) {
          for (auto &intri : cameras_all[id]) {
            primal_residual += (intri.params - cameras_avg[id].params).norm();
          }
        }
        
        // accumulate dual residual
        if (iter > 0) {
          dual_residual += (cameras_avg[id].params - prev_cameras_avg[id].params).norm();
        }
        
      }
      
      vector<double> errors;
      double max_error = 0;
      // get average of extrinsics
      for (auto &avg : images_avg) {
        image_t image_id = avg.first;
        p_extri_[extri_count++] = avg.second;
        
        class Image &image = reconstruction->Image(image_id);
        double error = (ProjectionCenterFromParameters(avg.second.qvec, avg.second.tvec) - image.TvecPrior()).norm();
        errors.push_back(error);
        max_error = error > max_error ? error : max_error;
        
        // accumulate primal residual
        if (images_all[image_id].size() > 1) {
          Eigen::Vector3d avg_angle_axis = Eigen::Vector3d::Constant(0);
          for (auto &extri : images_all[image_id]) {
            avg_angle_axis += QuaternionToAngleAxis(extri.qvec);
            primal_residual += (QuaternionToAngleAxis(extri.qvec) - QuaternionToAngleAxis(avg.second.qvec)).norm();
            primal_residual += (ProjectionCenterFromParameters(extri.qvec, extri.tvec) - ProjectionCenterFromParameters(avg.second.qvec, avg.second.tvec)).norm();
            cout << "single quaternion to angle axis : " << QuaternionToAngleAxis(extri.qvec) << endl;
          }
          avg_angle_axis /= images_all[image_id].size();
          cout << "avg_angle_axis: " << avg_angle_axis << endl;
          cout << "avg quaternion to angle axis: " << QuaternionToAngleAxis(avg.second.qvec) << endl;
        }
        
        // accumulate dual residual
        if (iter > 0) {
          Extrinsic prev_extri = prev_images_avg[image_id];
          dual_residual += (QuaternionToAngleAxis(prev_extri.qvec) - QuaternionToAngleAxis(avg.second.qvec)).norm();
          dual_residual += (ProjectionCenterFromParameters(prev_extri.qvec, prev_extri.tvec) - ProjectionCenterFromParameters(avg.second.qvec, avg.second.tvec)).norm();
        }
      }
     
      printf("Overall fitting error: %lu(poses), %lf(mean), %lf(median), %lf(max)", errors.size(), Mean(errors), Median(errors), max_error);
      printf("primal residual: %lf\n", primal_residual);
      printf("dual residual: %lf\n", dual_residual);
      printf("extrinsics number:%d root extrinsics:%zu", extri_count, reconstruction->NumImages());
      
      prev_images_avg = images_avg;
      prev_cameras_avg = cameras_avg;
      
    } else {
      
      // send intrinsics
      
      for (const camera_t camera_id : camera_ids_) {
        Camera& camera = reconstruction->Camera(camera_id);
        struct Intrinsic intri;
        intri.camera_id = camera_id;
        
        for (int i = 0; i < 4; i++) {
          intri.params[i] = camera.Params()[i];
        }
        
        p_intri_[intri_count++] = intri;
        
        cout << "process  " << rank << " send intrinsic focal length " << p_intri_[intri_count-1].params[0] << endl;
      }
      
      MPI_Send(p_intri_, sizeof(Intrinsic) * intri_count, MPI_BYTE, 0, rank, MPI_COMM_WORLD);
      
      // send extrinsics
      for (image_t image_id : config_.Images()) {
        class Image &image = reconstruction->Image(image_id);
        struct Extrinsic extrinsic;
        extrinsic.image_id = image_id;
        extrinsic.qvec = image.Qvec();
        extrinsic.tvec = image.Tvec();
        p_extri_[extri_count++] = extrinsic;
      }
      MPI_Send(p_extri_, sizeof(Extrinsic) * extri_count, MPI_BYTE, 0, rank, MPI_COMM_WORLD);
      printf("process %d sent data to root\n", rank);
    }
    
    // broadcast average intrinsics
    
    MPI_Bcast(&intri_count, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(p_intri_, intri_count * sizeof(Intrinsic), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    for (int j = 0; j < intri_count; j++) {
      Intrinsic avg = p_intri_[j];
      class Camera &camera = reconstruction->Camera(avg.camera_id);
      //update error
      for (int i = 0; i < 4; i++) {
        camera.params_error_[i] = camera.params_error_[i] + (1 + pow_alpha) * (camera.Params()[i] - avg.params[i]);
      }
      camera.SetParams({avg.params[0], avg.params[1], avg.params[2], avg.params[3]});
      
      cout << "process " << rank << " focal length " << avg.params[0] << " principal x " << avg.params[1] << " principal y " << avg.params[2] << " error " << camera.params_error_[0] << " " << camera.params_error_[1] << " " << camera.params_error_[2] << endl;
    }

    // broadcast average extrinsics
    MPI_Bcast(&extri_count, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(p_extri_, extri_count * sizeof(Extrinsic), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    for (int j = 0; j < extri_count; j++) {
      Extrinsic avg = p_extri_[j];
      if (!reconstruction->IsImageRegistered(avg.image_id)) continue;
      class Image &image = reconstruction->Image(avg.image_id);
      //update error
      image.qvec_error_ = image.qvec_error_ + image.Qvec() - avg.qvec;
      
      
      image.angle_error_ = image.angle_error_ + (1 + pow_alpha) * (QuaternionToAngleAxis(image.Qvec()) - QuaternionToAngleAxis(avg.qvec));
      
      //image.tvec_error_ = image.tvec_error_ + image.Tvec() - avg.tvec;
      image.tvec_error_ = image.tvec_error_ + (1 + pow_alpha) * (ProjectionCenterFromParameters(image.Qvec(), image.Tvec()) - ProjectionCenterFromParameters(avg.qvec, avg.tvec));
      
//      if (j < 10) {
//        cout << "angle error " << rank << " " << image.angle_error_ << endl;
//        cout << "center error " << rank << " " << image.tvec_error_ << endl;
//        cout << "process " << rank << " projection center " << image.ProjectionCenter() << " avg center " << ProjectionCenterFromParameters(avg.qvec, avg.tvec) << endl;
//      }

      image.SetQvec(avg.qvec);
      image.SetTvec(avg.tvec);
      
    }
    reconstruction->prev_points3D_ = reconstruction->points3D_;
    
    // self adaption step. It's easy to yield diverged results.
    // Not use at moment.
    for (int j = 0; j < 4; j++) {
      //ro_intri_[j] *= 0.1;
    }
    
    for (int j = 0; j < 2; j++) {
      //ro_extri_[j] *= 0.1;
    }
    
    for (int j = 0; j < 3; j++) {
      //ro_point_[j] *= 0.1;
    }
    pow_alpha *= options_.over_relaxation_alpha;
  }
  return true;
}

bool BundleAdjuster::Solve(Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);
  CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";

  point3D_num_images_.clear();

  problem_.reset(new ceres::Problem());

  ceres::LossFunction* loss_function = options_.CreateLossFunction();

  //ceres::LossFunction * loss_function = new ceres::HuberLoss(4.0*4.0); //no enough exps, commented temporarily
  SetUp(reconstruction, loss_function);

  if (problem_->NumResiduals() == 0) {
    return false;
  }
  Timer timer;
  timer.Start();

  ceres::Solver::Options solver_options = options_.solver_options;

  // Empirical choice.
  const size_t kMaxNumImagesDirectDenseSolver = 50;
  const size_t kMaxNumImagesDirectSparseSolver = 500;
  const size_t num_images = config_.NumImages();
  if (num_images <= kMaxNumImagesDirectDenseSolver) {
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (num_images <= kMaxNumImagesDirectSparseSolver) {
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {  // Indirect sparse (preconditioned CG) solver.
    solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

#ifdef OPENMP_ENABLED
  solver_options.num_threads =
       GetEffectiveNumThreads(solver_options.num_threads);
  solver_options.num_linear_solver_threads =
       GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif
  
//  cout << "solver_options.num_threads: " << solver_options.num_threads << endl;
//  cout << "solver_options.num_linear_solver_threads: " << solver_options.num_linear_solver_threads << endl;

  std::string error;
  CHECK(solver_options.IsValid(&error)) << error;

  double initial_reproj_error = 0;
  double initial_gps_error = 0;
  double initial_dba_intri = 0;
  double initial_dba_extri = 0;
  double initial_dba_point = 0;
  
  ceres::Problem::EvaluateOptions evaOptions;
  
  evaOptions.residual_blocks = pixel_block_ids_;
  problem_->Evaluate(evaOptions, &initial_reproj_error, nullptr, nullptr, nullptr);
  
  evaOptions.residual_blocks = gps_block_ids_;
  problem_->Evaluate(evaOptions, &initial_gps_error, nullptr, nullptr, nullptr);
  
  if (dba_used_) {
    evaOptions.residual_blocks = dba_intri_block_ids_;
    problem_->Evaluate(evaOptions, &initial_dba_intri, nullptr, nullptr, nullptr);
  
    evaOptions.residual_blocks = dba_extri_block_ids_;
    problem_->Evaluate(evaOptions, &initial_dba_extri, nullptr, nullptr, nullptr);
  
    evaOptions.residual_blocks = dba_point_block_ids_;
    problem_->Evaluate(evaOptions, &initial_dba_point, nullptr, nullptr, nullptr);
  }
  
  ceres::Solve(solver_options, problem_.get(), &summary_);
    std::cout << "ceres::Solve time cost " << StringPrintf("%.3fs", timer.ElapsedSeconds()) << std::endl;

  if (options_.use_angle_axis) {
    for (const image_t image_id : config_.Images()) {
      Image& image = reconstruction->Image(image_id);
      image.TransformAngleAxisToQuaternion();
  }
  }

  // Get reprojection error
  double final_reproj_error = 0;
  double final_gps_error = 0;
  double final_dba_intri = 0;
  double final_dba_extri = 0;
  double final_dba_point = 0;
  
  evaOptions.residual_blocks = pixel_block_ids_;
  problem_->Evaluate(evaOptions, &final_reproj_error, nullptr, nullptr, nullptr);
  
  evaOptions.residual_blocks = gps_block_ids_;
  problem_->Evaluate(evaOptions, &final_gps_error, nullptr, nullptr, nullptr);
  
  if (dba_used_) {
    evaOptions.residual_blocks = dba_intri_block_ids_;
    problem_->Evaluate(evaOptions, &final_dba_intri, nullptr, nullptr, nullptr);
    
    evaOptions.residual_blocks = dba_extri_block_ids_;
    problem_->Evaluate(evaOptions, &final_dba_extri, nullptr, nullptr, nullptr);
    
    evaOptions.residual_blocks = dba_point_block_ids_;
    problem_->Evaluate(evaOptions, &final_dba_point, nullptr, nullptr, nullptr);
  }
  
  //--
  //Error After BA
  // Collect corresponding camera centers
  cout << endl
            << "Bundle Adjustment statistics (approximated RMSE):\n"
            << " #views: " << reconstruction->NumImages() << "\n"
            << " #poses: " << config_.NumImages() << "\n"
            << " #cameras: " << reconstruction->NumCameras() << "\n"
            << " #Points3D: " << reconstruction->NumPoints3D() << "\n"
            << " #residuals: " << summary_.num_residuals << "\n"
            << " #reproj residuals: " << pixel_block_ids_.size() << "\n"
            << " Initial reproj error: " << std::sqrt(initial_reproj_error / pixel_block_ids_.size()) << "\n"
            << " Final reproj error: " << std::sqrt(final_reproj_error / pixel_block_ids_.size()) << "\n"
            << " #gps residuals: " << gps_block_ids_.size() << "\n"
            << " Initial gps error: " << std::sqrt(initial_gps_error / gps_block_ids_.size()) << "\n"
            << " Final gps error: " << std::sqrt(final_gps_error / gps_block_ids_.size()) << "\n";

  if (dba_used_) {
    cout << " #dba intri residuals: " << dba_intri_block_ids_.size() << "\n"
         << " Initial dba intri error: " << sqrt(initial_dba_intri / dba_intri_block_ids_.size()) << "\n"
         << " final dba intri error: " << sqrt(final_dba_intri / dba_intri_block_ids_.size()) << "\n"
    
         << " #dba extri residuals: " << dba_extri_block_ids_.size() << "\n"
         << " Initial dba extri error: " << sqrt(initial_dba_extri / dba_extri_block_ids_.size()) << "\n"
         << " Final dba extri error: " << sqrt(final_dba_extri / dba_extri_block_ids_.size()) << "\n"
    
         << " #dba point residuals: " << dba_point_block_ids_.size() << "\n"
         << " Initial dba point error: " << sqrt(initial_dba_point / dba_point_block_ids_.size()) << "\n"
         << " Final dba point error: " << sqrt(final_dba_point / dba_point_block_ids_.size()) << "\n";

  }
  
  cout << " Iterations: " << summary_.iterations.size() << "\n"
            << " Time (s): " << summary_.total_time_in_seconds << "\n\n"
            << endl;

  if (reconstruction->ecef_established_) {
    PrintErrorStatistics(reconstruction);
  }

  /*for (const auto Image_ : reconstruction->Images()) {
    if (reconstruction->IsImageRegistered(Image_.first)) {
      Image& image = reconstruction->Image(Image_.first);
      Eigen::Vector3d tmp_center(image.ProjectionCenter());
      //std::cout << "DEV: Image name: "<< image.Name() << '\n';

      //std::cout << StringPrintf("DEV: => Center: %f, %f, %f", tmp_center(0), tmp_center(1), tmp_center(2)) << std::endl;
    }
  }*/

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options_.print_summary) {
    PrintHeading2("Bundle adjustment report");
    PrintSolverSummary(summary_);
    
    //cout << "-----------------------" << endl;
    //cout << summary_.FullReport() << endl;
  }

  // 这里完成了计算，把模型平移回去
  if (reconstruction->centroid_moved_) {
    timer.Restart();
    // set back to the original scene centroid
    reconstruction->Transform(1.0, ComposeIdentityQuaternion(), reconstruction->pose_centroid_ * -1.0, &config_.Images());
    reconstruction->centroid_moved_ = false;
    std::cout << "Success Move Model Back!" << "\n";
    std::cout << "Model Back time cost " << StringPrintf("%.3fs", timer.ElapsedSeconds()) << std::endl;
  }

  TearDown(reconstruction);

  return true;
}

ceres::Solver::Summary BundleAdjuster::Summary() const { return summary_; }

//DEV: add a parm "Eigen::Vector3d"
void BundleAdjuster::SetUp(Reconstruction* reconstruction,
                           ceres::LossFunction* loss_function) {
  // options_.use_motion_priors_opt 为openMVG的-P选项
  // 在这里加入openMVG中相似变化的相关代码
  Timer timer;
  timer.Start();
  //openMVG::geometry::Similarity3 sim_to_center;
  if (options_.use_motion_priors_opt && config_.Images().size() > 3 && !dba_used_) {
    alignment_success_ = reconstruction->SimilarityTransform(options_.ransac_max_error, &config_.Images());
  }

  //在加入相机位置的误差时，
  //应该不需要新增加ceres变量，使用下面AddImageToProblem中用到的相机位置即可

  pixel_block_ids_.clear();
  gps_block_ids_.clear();
  
  if (dba_used_) {
    dba_intri_block_ids_.clear();
    dba_extri_block_ids_.clear();
    dba_point_block_ids_.clear();
  }
  
  // Warning: AddPointsToProblem assumes that AddImageToProblem is called first.
  // Do not change order of instructions!
  for (const image_t image_id : config_.Images()) {
    AddImageToProblem(image_id, reconstruction, loss_function);
  }
  
  // The following two for loops are useless at the moment.
  for (const auto point3D_id : config_.VariablePoints()) {
    //cout << "variable points found" << endl;
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }
  for (const auto point3D_id : config_.ConstantPoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
    cout << "constant points found" << endl;
  }


  std::cout << "Add Points projection Error as residual time cost " << StringPrintf("%.3fs", timer.ElapsedSeconds()) << std::endl;
  timer.Restart();

  // Add Pose prior constraints if any
  //std::map<image_t, std::vector<double> > map_poses;
  if (alignment_success_) {
    std::cout<< "Use prior constrait Images number: " << config_.Images().size() <<std::endl;
    /*size_t config_num_obs = 0;
    for (const image_t image_id : config_.Images()) {
      config_num_obs += reconstruction->Image(image_id).NumPoints3D();
    }*/
    //config_.Images().size() * 2.0 / 3.0;
    //std::cout << "GPS Prior Weight: " << gps_weight << std::endl;
    size_t residuals_reproj = problem_->NumResiduals();
    double gps_weight = options_.gps_weight; 

    std::cout << "Total residuals - GPS residual = " << residuals_reproj << " GPS weight = " << gps_weight << std::endl;

    for (const image_t image_id : config_.Images()) {
      Image& prior = reconstruction->Image(image_id);
       //判断图片是否有GPS的时候加上在原有的image.HasTvecPrior() 基础上加上 image.TvecPrior().sum() != 0
      if (prior.HasTvecPrior() && prior.TvecPrior().sum() != 0 && prior.HasCamera()
          && reconstruction->IsImageRegistered(image_id)) {

        double* qvec_data = prior.Qvec().data();
        double* tvec_data = prior.Tvec().data();
        double* procen_data = prior.ProjectionCenterPoint().data();
        //const Mat3 R = prior.rotation();

        //ceres::RotationMatrixToAngleAxis((const double*)R.data(), angleAxis);

        //map_poses[image_id] = {angleAxis[0], angleAxis[1], angleAxis[2], t[0], t[1], t[2]};
        // Add the cost functor (distance from Pose prior to the SfM_Data Pose center)

        //std::cout << "DEV: pro_center_: " << prior.ProjectionCenter() << "\n";
        //std::cout << "DEV: gps_center_: " << prior.TvecPrior() << "\n";
        //Eigen::Vector3d ecef_gps(lla_to_ecef(prior.TvecPrior(0), prior.TvecPrior(1), prior.TvecPrior(2)));
        ceres::CostFunction * cost_function = nullptr;

        if (options_.use_angle_axis) {
          cost_function =
            new ceres::AutoDiffCostFunction<PoseCenterConstraintProjectionCenterCostFunction, 3, 3>(
              new PoseCenterConstraintProjectionCenterCostFunction(prior.TvecPrior(), Eigen::Vector3d::Constant(options_.gps_weight)));
          double pose_center_robust_fitting_error = reconstruction->pose_center_robust_fitting_error_;
          gps_block_ids_.push_back(problem_->AddResidualBlock(cost_function, \
            new ceres::HuberLoss(pose_center_robust_fitting_error*pose_center_robust_fitting_error), \
            procen_data));
        }
        else {
          cost_function =
            new ceres::AutoDiffCostFunction<PoseCenterConstraintCostFunction, 3, 4, 3>(
              new PoseCenterConstraintCostFunction(prior.TvecPrior(), Eigen::Vector3d::Constant(options_.gps_weight)));
          double pose_center_robust_fitting_error = reconstruction->pose_center_robust_fitting_error_;
          gps_block_ids_.push_back(problem_->AddResidualBlock(cost_function, \
            new ceres::HuberLoss(pose_center_robust_fitting_error*pose_center_robust_fitting_error), \
            qvec_data, tvec_data));
        }
      }
    }
    std::cout << "Total residuals = " << problem_->NumResiduals() << std::endl;
    std::cout << "Add pose GPS error as residual time cost " << StringPrintf("%.3fs", timer.ElapsedSeconds()) << std::endl;
    timer.Restart();

  }

  if (dba_used_) {
    //cout << "use distribued" << endl;
    AddLagrangianToProblem(reconstruction, loss_function);
  }
  
  ParameterizeCameras(reconstruction);
  ParameterizePoints(reconstruction);
}

void BundleAdjuster::TearDown(Reconstruction*) {
  problem_.reset();
  // Nothing to do
}

void BundleAdjuster::AddImageToProblem(const image_t image_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
  Image& image = reconstruction->Image(image_id);
  Camera& camera = reconstruction->Camera(image.CameraId());

  // CostFunction assumes unit quaternions.
  image.NormalizeQvec();
  if (options_.use_angle_axis)   
    image.TransformQuaternionToAngleAxis();

  double* qvec_data = image.Qvec().data();
  double* tvec_data = image.Tvec().data();
  double* angaxis_data = image.AngleAxis().data();
  double* procen_data = image.ProjectionCenterPoint().data();
  double* camera_params_data = camera.ParamsData();

  // Collect cameras for final parameterization.
  CHECK(image.HasCamera());

  const bool constant_pose = config_.HasConstantPose(image_id);

  // Add residuals to bundle adjustment problem.
  size_t num_observations = 0;
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D()) {
      continue;
    }
    if (!reconstruction->ExistsPoint3D(point2D.Point3DId())) {
      continue;
    }

    num_observations += 1;
    point3D_num_images_[point2D.Point3DId()] += 1;

    Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());
    assert(point3D.Track().Length() > 1);

    ceres::CostFunction* cost_function = nullptr;

    if (constant_pose) {
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                    \
  case CameraModel::kModelId:                                             \
    if (options_.use_angle_axis) {                         \
      cost_function =                                                       \
        BundleAdjustmentProjectionCenterConstantPoseCostFunction<CameraModel>::Create(    \
            image.AngleAxis(), image.ProjectionCenterPoint(), point2D.XY());       \
        }   \
    else {  \
      cost_function =                                                       \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(    \
            image.Qvec(), image.Tvec(), point2D.XY());           \
        }  \
    pixel_block_ids_.push_back(problem_->AddResidualBlock(cost_function, loss_function,              \
                               point3D.XYZ().data(), camera_params_data)); \    
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
      }
    } else {
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                   \
  case CameraModel::kModelId:                                            \
    if (options_.use_angle_axis) {                         \
      cost_function =                                                      \
        BundleAdjustmentProjectionCenterCostFunction<CameraModel>::Create(point2D.XY()); \
      pixel_block_ids_.push_back(problem_->AddResidualBlock(cost_function, loss_function, angaxis_data,  \
                               procen_data, point3D.XYZ().data(),          \
                               camera_params_data));     \
      }                      \
    else {        \
      cost_function =                                                      \
        BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
      pixel_block_ids_.push_back(problem_->AddResidualBlock(cost_function, loss_function, qvec_data,  \
                               tvec_data, point3D.XYZ().data(),          \
                               camera_params_data));                      \
      }  \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
      }
    }
  }

  if (num_observations > 0) {
    camera_ids_.insert(image.CameraId());

    // Set pose parameterization.
    if (!constant_pose) {
      if (options_.use_angle_axis) 
      {
        ceres::LocalParameterization* angaxis_parameterization =
          new ceres::IdentityParameterization(3);
        problem_->SetParameterization(angaxis_data, angaxis_parameterization);
        if (config_.HasConstantTvec(image_id)) {
          const std::vector<int>& constant_tvec_idxs =
              config_.ConstantTvec(image_id);
          ceres::SubsetParameterization* procen_parameterization =
              new ceres::SubsetParameterization(3, constant_tvec_idxs);
          problem_->SetParameterization(procen_data, procen_parameterization);
        }
      }
      else
      {
        ceres::LocalParameterization* quaternion_parameterization =
          new ceres::QuaternionParameterization;
        problem_->SetParameterization(qvec_data, quaternion_parameterization);
        if (config_.HasConstantTvec(image_id)) {
          const std::vector<int>& constant_tvec_idxs =
              config_.ConstantTvec(image_id);
          ceres::SubsetParameterization* tvec_parameterization =
              new ceres::SubsetParameterization(3, constant_tvec_idxs);
          problem_->SetParameterization(tvec_data, tvec_parameterization);
        }
      }      
    }
  }
}

void BundleAdjuster::AddPointToProblem(const point3D_t point3D_id,
                                       Reconstruction* reconstruction,
                                       ceres::LossFunction* loss_function) {
  Point3D& point3D = reconstruction->Point3D(point3D_id);

  // Is 3D point already fully contained in the problem? I.e. its entire track
  // is contained in `variable_image_ids`, `constant_image_ids`,
  // `constant_x_image_ids`.
  if (point3D_num_images_[point3D_id] == point3D.Track().Length()) {
    return;
  }

  for (const auto& track_el : point3D.Track().Elements()) {
    // Skip observations that were already added in `FillImages`.
    if (config_.HasImage(track_el.image_id)) {
      continue;
    }

    point3D_num_images_[point3D_id] += 1;

    Image& image = reconstruction->Image(track_el.image_id);
    Camera& camera = reconstruction->Camera(image.CameraId());
    const Point2D& point2D = image.Point2D(track_el.point2D_idx);

    // We do not want to refine the camera of images that are not
    // part of `constant_image_ids_`, `constant_image_ids_`,
    // `constant_x_image_ids_`.
    if (camera_ids_.count(image.CameraId()) == 0) {
      camera_ids_.insert(image.CameraId());
      config_.SetConstantCamera(image.CameraId());
    }

    ceres::CostFunction* cost_function = nullptr;

    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                     \
  case CameraModel::kModelId:                                              \
    cost_function =                                                        \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(     \
            image.Qvec(), image.Tvec(), point2D.XY());                     \
    problem_->AddResidualBlock(cost_function, loss_function,               \
                               point3D.XYZ().data(), camera.ParamsData()); \
    break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }
  }
}

void BundleAdjuster::AddLagrangianExtrinsicsToProblem(Reconstruction* reconstruction, ceres::LossFunction* loss_function)
{
  for (const image_t image_id : config_.Images()) {
    Image& image = reconstruction->Image(image_id);
    ceres::CostFunction * cost_function =
        new ceres::AutoDiffCostFunction<LagrangianExtrinsicsConstraintCostFunction, 6, 4, 3>(
            new LagrangianExtrinsicsConstraintCostFunction(QuaternionToAngleAxis(image.Qvec()), ProjectionCenterFromParameters(image.Qvec(), image.Tvec()), image.angle_error_, image.tvec_error_, ro_extri_));
    
    dba_extri_block_ids_.push_back(problem_->AddResidualBlock(cost_function, loss_function, image.Qvec().data(), image.Tvec().data()));
  }
}

void BundleAdjuster::AddLagrangianPointToProblem(Reconstruction* reconstruction, ceres::LossFunction* loss_function)
{
  std::unordered_set<point3D_t> added;
  for (const image_t image_id : config_.Images()) {
    Image& image = reconstruction->Image(image_id);
    
    // CostFunction assumes unit quaternions.
    image.NormalizeQvec();
    
    // Collect cameras for final parameterization.
    CHECK(image.HasCamera());
    
    // Add residuals to bundle adjustment problem.
    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D()) {
        continue;
      }
      if (!reconstruction->ExistsPoint3D(point2D.Point3DId())) {
        continue;
      }
      if (added.find(point2D.Point3DId()) != added.end()) {
        continue;
      }
      
      Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());
      assert(point3D.Track().Length() > 1);
      
      ceres::CostFunction * cost_function =
          new ceres::AutoDiffCostFunction<LagrangianPointConstraintCostFunction, 3, 3>(
              new LagrangianPointConstraintCostFunction(point3D.XYZ(), ro_point_));
      
      dba_point_block_ids_.push_back(problem_->AddResidualBlock(cost_function, loss_function, point3D.XYZ().data()));
      added.insert(point2D.Point3DId());
    }
  }
}

void BundleAdjuster::AddLagrangianToProblem(Reconstruction* reconstruction, ceres::LossFunction* loss_function)
{
  const bool constant_camera = !options_.refine_focal_length &&
  !options_.refine_principal_point &&
  !options_.refine_extra_params;
  for (const camera_t camera_id : camera_ids_) {
    Camera& camera = reconstruction->Camera(camera_id);
    
    if (constant_camera || config_.IsConstantCamera(camera_id)) {
      continue;
    } else {
      
      //printf("%s: debug intrinsic %lf %lf %lf %lf\n", __func__, camera.FocalLength(), camera.ParamsData()[1], camera.ParamsData()[2], camera.ParamsData()[3]);
      ceres::CostFunction * cost_function =
      new ceres::AutoDiffCostFunction<LagrangianIntrinsicsConstraintCostFunction, 4, SimpleRadialCameraModel::kNumParams>(
          new LagrangianIntrinsicsConstraintCostFunction(camera, reconstruction->Camera(camera_id).params_error_, ro_intri_));
      //printf("cost function built\n");
      //printf("size of paramsdata:%lu\n", camera.Params().size());
      
      dba_intri_block_ids_.push_back(problem_->AddResidualBlock(cost_function, loss_function, camera.ParamsData()));
    }
  }
  
  AddLagrangianExtrinsicsToProblem(reconstruction, loss_function);
  AddLagrangianPointToProblem(reconstruction, loss_function);
}

void BundleAdjuster::ParameterizeCameras(Reconstruction* reconstruction) {
  const bool constant_camera = !options_.refine_focal_length &&
                               !options_.refine_principal_point &&
                               !options_.refine_extra_params;
  for (const camera_t camera_id : camera_ids_) {
    Camera& camera = reconstruction->Camera(camera_id);

    if (constant_camera || config_.IsConstantCamera(camera_id)) {
      problem_->SetParameterBlockConstant(camera.ParamsData());
      continue;
    } else {
      std::vector<int> const_camera_params;

      if (!options_.refine_focal_length) {
        const std::vector<size_t>& params_idxs = camera.FocalLengthIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }
      if (!options_.refine_principal_point) {
        const std::vector<size_t>& params_idxs = camera.PrincipalPointIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }
      if (!options_.refine_extra_params) {
        const std::vector<size_t>& params_idxs = camera.ExtraParamsIdxs();
        const_camera_params.insert(const_camera_params.end(),
                                   params_idxs.begin(), params_idxs.end());
      }

      if (const_camera_params.size() > 0) {
        ceres::SubsetParameterization* camera_params_parameterization =
            new ceres::SubsetParameterization(
                static_cast<int>(camera.NumParams()), const_camera_params);
        problem_->SetParameterization(camera.ParamsData(),
                                      camera_params_parameterization);
      }
    }
  }
}

void BundleAdjuster::ParameterizePoints(Reconstruction* reconstruction) {
  for (const auto num_images : point3D_num_images_) {
    if (!config_.HasVariablePoint(num_images.first) &&
        !config_.HasConstantPoint(num_images.first)) {
      Point3D& point3D = reconstruction->Point3D(num_images.first);
      if (point3D.Track().Length() > point3D_num_images_[num_images.first]) {
        problem_->SetParameterBlockConstant(point3D.XYZ().data());
      }
    }
  }

  for (const point3D_t point3D_id : config_.ConstantPoints()) {
    Point3D& point3D = reconstruction->Point3D(point3D_id);
    problem_->SetParameterBlockConstant(point3D.XYZ().data());
  }
}

DistributedBundleAdjuster::DistributedBundleAdjuster(const BundleAdjuster::Options& options)
  : options_(options) {
  CHECK(options_.Check());
}

void DistributedBundleAdjuster::UpdateIntrinsics(Reconstruction* recon, EIGEN_STL_UMAP(image_t, Intrinsic) &cameras_avg) {
  CHECK_EQ(recon->Cameras().size(), cameras_avg.size());
  for (auto &id_camera : recon->Cameras()) {
    //CHECK_NE(cameras_avg.find(id_camera.first), cameras_avg.end());
    Intrinsic &avg = cameras_avg[id_camera.first];
    class Camera &camera = recon->Camera(id_camera.first);
    camera.SetParams({avg.params[0], avg.params[1], avg.params[2], avg.params[3]});
  }
}

void DistributedBundleAdjuster::UpdateExtrinsics(Reconstruction* recon, EIGEN_STL_UMAP(image_t, Extrinsic) &images_avg) {
  for (auto &id : recon->RegImageIds()) {
    if (images_avg.find(id) == images_avg.end()) continue;
    Extrinsic avg = images_avg[id];
    class Image &image = recon->Image(avg.image_id);
    image.SetQvec(avg.qvec);
    image.SetTvec(avg.tvec);
  }
}

void DistributedBundleAdjuster::UpdatePoints(Reconstruction* all, Reconstruction* sub)
{
  for (auto &id : sub->Point3DIds()) {
    Point3D &src = sub->Point3D(id);
    CHECK_EQ(all->ExistsPoint3D(id), true);
    Point3D &dst = all->Point3D(id);
    dst.SetXYZ(src.XYZ());
  }
}

bool DistributedBundleAdjuster::Solve(Reconstruction* reconstruction) {
  
  int nprocs = options_.dba_threads;
  int block_images = options_.dba_block_images;
  
  if (options_.dba_use_dynamic_threads) {
    nprocs = nprocs > reconstruction->NumRegImages() / block_images + 1 ? nprocs :
        reconstruction->NumRegImages() / block_images + 1;
  }
  
  bool alignment_success = false;
  if (options_.use_motion_priors_opt && reconstruction->NumRegImages() > 3) {
    alignment_success = reconstruction->SimilarityTransform(options_.ransac_max_error);
  }
  std::vector<std::unique_ptr<Reconstruction>> recons = reconstruction->Split(nprocs);
  
  vector<std::unique_ptr<BundleAdjuster>> adjusters;
  for (int pid = 0; pid < nprocs; pid++) {

    Reconstruction *recon = recons[pid].get();
    CHECK_NOTNULL(reconstruction);

    const std::vector<image_t>& reg_image_ids = recon->RegImageIds();

    CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
    "registered for global "
    "bundle-adjustment";
    
    // Configure bundle adjustment.
    BundleAdjustmentConfig ba_config;
    for (const image_t image_id : reg_image_ids) {
      ba_config.AddImage(image_id);
    }
    ba_config.SetConstantPose(reg_image_ids[0]);
    ba_config.SetConstantTvec(reg_image_ids[1], {0});
    
    for (const image_t image_id : ba_config.Images()) {
      class Image &image = recon->Image(image_id);
      image.qvec_error_ = Eigen::Vector4d::Constant(0);
      image.angle_error_ = Eigen::Vector3d::Constant(0);
      image.tvec_error_ = Eigen::Vector3d::Constant(0);
    }
  
    adjusters.emplace_back(new BundleAdjuster(options_, ba_config));
    adjusters[pid].get()->InitDBAParameters(recon);
    adjusters[pid].get()->dba_used_ = true;
    adjusters[pid].get()->alignment_success_ = alignment_success;
  }
  
  EIGEN_STL_UMAP(image_t, Extrinsic) prev_images_avg;
  EIGEN_STL_UMAP(camera_t, Intrinsic) prev_cameras_avg;
  
  EIGEN_STL_UMAP(image_t, int) images_count;
  EIGEN_STL_UMAP(image_t, Extrinsic) images_avg;
  EIGEN_STL_UMAP(image_t, vector<Extrinsic>) images_all;
  
  EIGEN_STL_UMAP(camera_t, int) cameras_count;
  EIGEN_STL_UMAP(camera_t, Intrinsic) cameras_avg;
  EIGEN_STL_UMAP(camera_t, vector<Intrinsic>) cameras_all;
  
  double pow_alpha = 1;
  
  printf("DBA %d(iter) ", 0);
  reconstruction->PrintSummary();
  
  int extra_iters = 0;
  for (int iter = 1; iter <= options_.dba_max_iterations + extra_iters && iter <= 30; iter++) {
   
    // print parameters
    for (int pid = 0; pid < nprocs; pid++) {
      BundleAdjuster* adjuster = adjusters[pid].get();
      Reconstruction *recon = recons[pid].get();
      cout << "process " << pid << " ro intri " << adjuster->ro_intri_[0] << " " << adjuster->ro_intri_[3] << " ro extri " << adjuster->ro_extri_[0] << " " << adjuster->ro_extri_[1] << " ro point " << adjuster->ro_point_[0] << endl;
      recon->PrintSummary();
    }

    reconstruction->FilterAllPoints3D(options_.dba_filter_max_reproj_error,
        options_.dba_filter_min_tri_angle,
        options_.dba_max_fitting_error);
    
    // solve parallel
#pragma omp parallel for
    for (int pid = 0; pid < nprocs; pid++) {
      //BundleAdjuster* adjuster = adjusters[pid].get();
      //Reconstruction *recon = recons[pid].get();
      
      // Avoid degeneracies in bundle adjustment.
      //recons[pid].get()->FilterObservationsWithNegativeDepth();
      recons[pid].get()->FilterAllPoints3D(options_.dba_filter_max_reproj_error,
          options_.dba_filter_min_tri_angle,
          options_.dba_max_fitting_error);
      adjusters[pid].get()->Solve(recons[pid].get());
    }

    double rotation_primal_residual = 0;
    double center_primal_residual = 0;
    double primal_residual = 0;
    double dual_residual = 0;
    double rotation_dual_residual = 0;
    double center_dual_residual = 0;
    double dual_points_residual = 0;
   
    images_count.clear();
    images_avg.clear();
    images_all.clear();
   
    cameras_count.clear();
    cameras_avg.clear();
    cameras_all.clear();
   
    for (int pid = 0; pid < nprocs; pid++) {
      BundleAdjuster* adjuster = adjusters[pid].get();
      Reconstruction *recon = recons[pid].get();

      if (iter > 0) {
        for (auto &point : recon->points3D_) {
         
          dual_points_residual += adjuster->ro_point_[0] * adjuster->ro_point_[0] * (point.second.XYZ() - recon->prev_points3D_[point.first].XYZ()).norm();
        }
      }

      // collect intrinsics

      for (const camera_t id : adjuster->CameraIds()) {
        Camera& camera = recon->Camera(id);
        struct Intrinsic intri;
        intri.camera_id = id;
       
        for (int i = 0; i < 4; i++) {
          intri.params[i] = camera.Params()[i];
        }
       
        if (cameras_count.find(id) == cameras_count.end()) {
          cameras_count[id] = 1;
          cameras_avg[id] = intri;
        } else {
          cameras_count[id]++;
          cameras_avg[id].params += intri.params;
        }
        cameras_all[id].push_back(intri);
        cout << "intrinsic from process " << pid << " focal length " <<intri.params[0] << " focal error: " << camera.params_error_[0] << endl;
      }

      // collect and average extrinsics

      for (image_t id : adjuster->Images()) {
        class Image &image = recon->Image(id);
        struct Extrinsic extri;
        extri.image_id = id;
        extri.qvec = image.Qvec();
        extri.tvec = image.Tvec();
       
        if (images_count.find(id) == images_count.end()) {
          images_count[id] = 1;
          images_avg[id] = extri;
        } else {
          images_count[id]++;
          Extrinsic &avg = images_avg[id];
          Eigen::Vector3d old_center = ProjectionCenterFromParameters(avg.qvec, avg.tvec);
          Eigen::Vector3d new_center = ProjectionCenterFromParameters(extri.qvec, extri.tvec);
          Eigen::Vector3d avg_center = (old_center * (images_count[id] - 1.0) + new_center) / images_count[id];
          //cout << "projection center before interpolation " << old_center << endl;
          //cout << "avg tvec " << avg.tvec << endl;
          //cout << "projection center of new " << new_center << endl;
          //cout << "new tvec " << p_extri[j].tvec << endl;
         
          avg.qvec = AverageQuaternions({avg.qvec, extri.qvec}, {double(images_count[id] - 1), 1.0});
          //avg.tvec = (avg.tvec * (images_count[image_id] - 1.0) + p_extri[j].tvec) / double(images_count[image_id]);
          avg.tvec = -QuaternionToRotationMatrix(avg.qvec) * avg_center;
        }
        images_all[id].push_back(extri);
      }
    }

    // get average of intrinsics
    
    for (auto &avg : cameras_avg) {
      camera_t id = avg.first;
      cameras_avg[id].params /= cameras_count[id];
      cout << "root get intrinsics average " << " focal length " << avg.second.params[0] << " principal x " << avg.second.params[1] << " principal y " << avg.second.params[2] << endl;
      
      // accumulate primal residual
      if (cameras_count[id] > 1) {
        for (auto &intri : cameras_all[id]) {
          primal_residual += (intri.params - cameras_avg[id].params).norm();
        }
      }
      
      // accumulate dual residual
      if (iter > 0) {
        dual_residual += (cameras_avg[id].params - prev_cameras_avg[id].params).norm();
      }
    }

    // update intrinsics and extrinsics
    for (int pid = 0; pid < nprocs; pid++) {
      Reconstruction *recon = recons[pid].get();
      
      // update intrinsics error
      for (auto &id_camera : recon->Cameras()) {
        Intrinsic &avg = cameras_avg[id_camera.first];
        class Camera &camera = recon->Camera(id_camera.first);
        
        //update error
        for (int i = 0; i < 4; i++) {
          camera.params_error_[i] = camera.params_error_[i] + (1 + pow_alpha) * (camera.Params()[i] - avg.params[i]);
        }
      }
      
      // update extrinsics error
      for (auto &id : recon->RegImageIds()) {
        Extrinsic avg = images_avg[id];
        class Image &image = recon->Image(avg.image_id);
        
        image.qvec_error_ = image.qvec_error_ + image.Qvec() - avg.qvec;
        
        image.angle_error_ = image.angle_error_ + (1 + pow_alpha) * (QuaternionToAngleAxis(image.Qvec()) - QuaternionToAngleAxis(avg.qvec));
        
        //image.tvec_error_ = image.tvec_error_ + image.Tvec() - avg.tvec;
        image.tvec_error_ = image.tvec_error_ + (1 + pow_alpha) * (ProjectionCenterFromParameters(image.Qvec(), image.Tvec()) - ProjectionCenterFromParameters(avg.qvec, avg.tvec));
        
      }

      recon->prev_points3D_ = recon->points3D_;
      
      UpdateIntrinsics(recon, cameras_avg);
      UpdateExtrinsics(recon, images_avg);
    }
    
    for (auto &avg : images_avg) {
      image_t image_id = avg.first;
      // accumulate primal residual
      if (images_all[image_id].size() > 1) {
        Eigen::Vector3d avg_angle_axis = Eigen::Vector3d::Constant(0);
        for (auto &extri : images_all[image_id]) {
          avg_angle_axis += QuaternionToAngleAxis(extri.qvec);
          double residual = (QuaternionToAngleAxis(extri.qvec) - QuaternionToAngleAxis(avg.second.qvec)).norm();
          primal_residual += residual;
          rotation_primal_residual += residual;
          residual = (ProjectionCenterFromParameters(extri.qvec, extri.tvec) - ProjectionCenterFromParameters(avg.second.qvec, avg.second.tvec)).norm();
          primal_residual += residual;
          center_primal_residual += residual;
          
          //cout << "single quaternion to angle axis : " << QuaternionToAngleAxis(extri.qvec) << endl;
        }
        avg_angle_axis /= images_all[image_id].size();
        //cout << "avg_angle_axis: " << avg_angle_axis << endl;
        //cout << "avg quaternion to angle axis: " << QuaternionToAngleAxis(avg.second.qvec) << endl;
      }
   
      // accumulate dual residual
      if (iter > 0) {
        Extrinsic prev_extri = prev_images_avg[image_id];
        double residual = (QuaternionToAngleAxis(prev_extri.qvec) - QuaternionToAngleAxis(avg.second.qvec)).norm();
        dual_residual += residual;
        rotation_dual_residual += residual;
        
        residual = (ProjectionCenterFromParameters(prev_extri.qvec, prev_extri.tvec) - ProjectionCenterFromParameters(avg.second.qvec, avg.second.tvec)).norm();
        dual_residual += residual;
        center_dual_residual += residual;
      }
    }
    
    prev_images_avg = images_avg;
    prev_cameras_avg = cameras_avg;
    
    pow_alpha *= options_.over_relaxation_alpha;
    
    UpdateIntrinsics(reconstruction, cameras_avg);
    //CHECK_EQ(reconstruction->RegImageIds().size(), images_avg.size());
    UpdateExtrinsics(reconstruction, images_avg);
    for (auto &recon : recons) {
      UpdatePoints(reconstruction, recon.get());
    }
    
    printf("primal residual: %lf\n", primal_residual);
    printf("rotation primal residual: %lf\n", rotation_primal_residual);
    printf("center primal residual: %lf\n", center_primal_residual);
    printf("dual residual: %lf\n", dual_residual);
    printf("rotation dual residual: %lf\n", rotation_dual_residual);
    printf("center dual residual: %lf\n", center_dual_residual);
    printf("points dual residual: %lf\n", dual_points_residual);
    
    printf("DBA %d(iter) ", iter);
    reconstruction->PrintSummary();
    
    double max_fitting_error = reconstruction->MaxFittingError();
    if (options_.dba_use_dynamic_iters &&
        max_fitting_error > 5 && (iter % options_.dba_max_iterations) == 0) {
      extra_iters += options_.dba_max_iterations;
    }
  }
  
  // 这里完成了计算，把模型平移回去
  if (reconstruction->centroid_moved_) {
    
    // set back to the original scene centroid
    reconstruction->Transform(1.0, ComposeIdentityQuaternion(), reconstruction->pose_centroid_ * -1.0);
    reconstruction->centroid_moved_ = false;
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// ParallelBundleAdjuster
////////////////////////////////////////////////////////////////////////////////



bool ParallelBundleAdjuster::Options::Check() const {
  CHECK_OPTION_GE(max_num_iterations, 0);
  return true;
}

ParallelBundleAdjuster::ParallelBundleAdjuster(
    const Options& options, const BundleAdjustmentConfig& config)
    : options_(options), config_(config), num_measurements_(0) {
  CHECK(options_.Check());
  CHECK(config_.NumConstantTvecs() == 0)
      << "PBA does not allow to set individual translational elements constant";
  CHECK(config_.NumVariablePoints() == 0 && config_.NumConstantPoints() == 0)
      << "PBA does not allow to parameterize individual 3D points";
}

bool ParallelBundleAdjuster::Solve(Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);
  CHECK_EQ(num_measurements_, 0)
      << "Cannot use the same ParallelBundleAdjuster multiple times";

  SetUp(reconstruction);

  pba::ParallelBA::DeviceT device;
  const size_t kMaxNumResidualsFloat = 100 * 1000;
  if (config_.NumResiduals(*reconstruction) > kMaxNumResidualsFloat) {
    // The threshold for using double precision is empirically chosen and
    // ensures that the system can be reliable solved.
    device = pba::ParallelBA::PBA_CPU_DOUBLE;
  } else {
    if (options_.gpu_index < 0) {
      device = pba::ParallelBA::PBA_CUDA_DEVICE_DEFAULT;
    } else {
      device = static_cast<pba::ParallelBA::DeviceT>(
          pba::ParallelBA::PBA_CUDA_DEVICE0 + options_.gpu_index);
    }
  }

  pba::ParallelBA pba(device, options_.num_threads);
  pba.SetNextBundleMode(pba::ParallelBA::BUNDLE_FULL);
  pba.EnableRadialDistortion(pba::ParallelBA::PBA_PROJECTION_DISTORTION);

  pba::ConfigBA* pba_config = pba.GetInternalConfig();
  pba_config->__lm_delta_threshold /= 100.0f;
  pba_config->__lm_gradient_threshold /= 100.0f;
  pba_config->__lm_mse_threshold = 0.0f;
  pba_config->__cg_min_iteration = 10;
  pba_config->__verbose_level = 2;
  pba_config->__lm_max_iteration = options_.max_num_iterations;

  pba.SetCameraData(cameras_.size(), cameras_.data());
  pba.SetPointData(points3D_.size(), points3D_.data());
  pba.SetProjection(measurements_.size(), measurements_.data(),
                    point3D_idxs_.data(), camera_idxs_.data());

  Timer timer;
  timer.Start();
  pba.RunBundleAdjustment();
  timer.Pause();

  // Compose Ceres solver summary from PBA options.
  summary_.num_residuals_reduced = static_cast<int>(2 * measurements_.size());
  summary_.num_effective_parameters_reduced =
      static_cast<int>(8 * config_.NumImages() -
                       2 * config_.NumConstantCameras() + 3 * points3D_.size());
  summary_.num_successful_steps = pba_config->GetIterationsLM() + 1;
  summary_.termination_type = ceres::TerminationType::USER_SUCCESS;
  summary_.initial_cost =
      pba_config->GetInitialMSE() * summary_.num_residuals_reduced / 4;
  summary_.final_cost =
      pba_config->GetFinalMSE() * summary_.num_residuals_reduced / 4;
  summary_.total_time_in_seconds = timer.ElapsedSeconds();

  TearDown(reconstruction);

  if (options_.print_summary) {
    PrintHeading2("Bundle adjustment report");
    PrintSolverSummary(summary_);
  }

  return true;
}

ceres::Solver::Summary ParallelBundleAdjuster::Summary() const {
  return summary_;
}

bool ParallelBundleAdjuster::IsReconstructionSupported(
    const Reconstruction& reconstruction) {
  std::set<camera_t> camera_ids;
  for (const auto& image : reconstruction.Images()) {
    if (image.second.IsRegistered()) {
      if (camera_ids.count(image.second.CameraId()) != 0 ||
          reconstruction.Camera(image.second.CameraId()).ModelId() !=
              SimpleRadialCameraModel::model_id) {
        return false;
      }
      camera_ids.insert(image.second.CameraId());
    }
  }
  return true;
}

void ParallelBundleAdjuster::SetUp(Reconstruction* reconstruction) {
  // Important: PBA requires the track of 3D points to be stored
  // contiguously, i.e. the point3D_idxs_ vector contains consecutive indices.
  cameras_.reserve(config_.NumImages());
  camera_ids_.reserve(config_.NumImages());
  ordered_image_ids_.reserve(config_.NumImages());
  image_id_to_camera_idx_.reserve(config_.NumImages());
  AddImagesToProblem(reconstruction);
  AddPointsToProblem(reconstruction);
}

void ParallelBundleAdjuster::TearDown(Reconstruction* reconstruction) {
  for (size_t i = 0; i < cameras_.size(); ++i) {
    const image_t image_id = ordered_image_ids_[i];
    const pba::CameraT& pba_camera = cameras_[i];

    // Note: Do not use PBA's quaternion methods as they seem to lead to
    // numerical instability or other issues.
    Image& image = reconstruction->Image(image_id);
    Eigen::Matrix3d rotation_matrix;
    pba_camera.GetMatrixRotation(rotation_matrix.data());
    pba_camera.GetTranslation(image.Tvec().data());
    image.Qvec() = RotationMatrixToQuaternion(rotation_matrix.transpose());

    Camera& camera = reconstruction->Camera(image.CameraId());
    camera.Params(0) = pba_camera.GetFocalLength();
    camera.Params(3) = pba_camera.GetProjectionDistortion();
  }

  for (size_t i = 0; i < points3D_.size(); ++i) {
    Point3D& point3D = reconstruction->Point3D(ordered_point3D_ids_[i]);
    points3D_[i].GetPoint(point3D.XYZ().data());
  }
}

void ParallelBundleAdjuster::AddImagesToProblem(
    Reconstruction* reconstruction) {
  for (const image_t image_id : config_.Images()) {
    const Image& image = reconstruction->Image(image_id);
    CHECK_EQ(camera_ids_.count(image.CameraId()), 0)
        << "PBA does not support shared intrinsics";

    const Camera& camera = reconstruction->Camera(image.CameraId());
    CHECK_EQ(camera.ModelId(), SimpleRadialCameraModel::model_id)
        << "PBA only supports the SIMPLE_RADIAL camera model";

    // Note: Do not use PBA's quaternion methods as they seem to lead to
    // numerical instability or other issues.
    const Eigen::Matrix3d rotation_matrix =
        QuaternionToRotationMatrix(image.Qvec()).transpose();

    pba::CameraT pba_camera;
    pba_camera.SetFocalLength(camera.Params(0));
    pba_camera.SetProjectionDistortion(camera.Params(3));
    pba_camera.SetMatrixRotation(rotation_matrix.data());
    pba_camera.SetTranslation(image.Tvec().data());

    CHECK(!config_.HasConstantTvec(image_id))
        << "PBA cannot fix partial extrinsics";
    if (config_.HasConstantPose(image_id)) {
      CHECK(config_.IsConstantCamera(image.CameraId()))
          << "PBA cannot fix extrinsics only";
      pba_camera.SetConstantCamera();
    } else if (config_.IsConstantCamera(image.CameraId())) {
      pba_camera.SetFixedIntrinsic();
    } else {
      pba_camera.SetVariableCamera();
    }

    num_measurements_ += image.NumPoints3D();
    cameras_.push_back(pba_camera);
    camera_ids_.insert(image.CameraId());
    ordered_image_ids_.push_back(image_id);
    image_id_to_camera_idx_.emplace(image_id,
                                    static_cast<int>(cameras_.size()) - 1);

    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D()) {
        point3D_ids_.insert(point2D.Point3DId());
      }
    }
  }
}

void ParallelBundleAdjuster::AddPointsToProblem(
    Reconstruction* reconstruction) {
  points3D_.resize(point3D_ids_.size());
  ordered_point3D_ids_.resize(point3D_ids_.size());
  measurements_.resize(num_measurements_);
  camera_idxs_.resize(num_measurements_);
  point3D_idxs_.resize(num_measurements_);

  int point3D_idx = 0;
  size_t measurement_idx = 0;

  for (const auto point3D_id : point3D_ids_) {
    const Point3D& point3D = reconstruction->Point3D(point3D_id);
    points3D_[point3D_idx].SetPoint(point3D.XYZ().data());
    ordered_point3D_ids_[point3D_idx] = point3D_id;

    for (const auto track_el : point3D.Track().Elements()) {
      if (image_id_to_camera_idx_.count(track_el.image_id) > 0) {
        const Image& image = reconstruction->Image(track_el.image_id);
        const Camera& camera = reconstruction->Camera(image.CameraId());
        const Point2D& point2D = image.Point2D(track_el.point2D_idx);
        measurements_[measurement_idx].SetPoint2D(
            point2D.X() - camera.Params(1), point2D.Y() - camera.Params(2));
        camera_idxs_[measurement_idx] =
            image_id_to_camera_idx_.at(track_el.image_id);
        point3D_idxs_[measurement_idx] = point3D_idx;
        measurement_idx += 1;
      }
    }
    point3D_idx += 1;
  }

  CHECK_EQ(point3D_idx, points3D_.size());
  CHECK_EQ(measurement_idx, measurements_.size());
}

////////////////////////////////////////////////////////////////////////////////
// RigBundleAdjuster
////////////////////////////////////////////////////////////////////////////////

RigBundleAdjuster::RigBundleAdjuster(const Options& options,
                                     const RigOptions& rig_options,
                                     const BundleAdjustmentConfig& config)
    : BundleAdjuster(options, config), rig_options_(rig_options) {}

bool RigBundleAdjuster::Solve(Reconstruction* reconstruction,
                              std::vector<CameraRig>* camera_rigs) {
  CHECK_NOTNULL(reconstruction);
  CHECK_NOTNULL(camera_rigs);
  CHECK(!problem_) << "Cannot use the same BundleAdjuster multiple times";

  // Check the validity of the provided camera rigs.
  std::unordered_set<camera_t> rig_camera_ids;
  for (auto& camera_rig : *camera_rigs) {
    camera_rig.Check(*reconstruction);
    for (const auto& camera_id : camera_rig.GetCameraIds()) {
      CHECK_EQ(rig_camera_ids.count(camera_id), 0)
          << "Camera must not be part of multiple camera rigs";
      rig_camera_ids.insert(camera_id);
    }

    for (const auto& snapshot : camera_rig.Snapshots()) {
      for (const auto& image_id : snapshot) {
        CHECK_EQ(image_id_to_camera_rig_.count(image_id), 0)
            << "Image must not be part of multiple camera rigs";
        image_id_to_camera_rig_.emplace(image_id, &camera_rig);
      }
    }
  }

  point3D_num_images_.clear();

  problem_.reset(new ceres::Problem());

  ceres::LossFunction* loss_function = options_.CreateLossFunction();
  SetUp(reconstruction, camera_rigs, loss_function);

  if (problem_->NumResiduals() == 0) {
    return false;
  }

  ceres::Solver::Options solver_options = options_.solver_options;

  // Empirical choice.
  const size_t kMaxNumImagesDirectDenseSolver = 50;
  const size_t kMaxNumImagesDirectSparseSolver = 1000;
  const size_t num_images = config_.NumImages();
  if (num_images <= kMaxNumImagesDirectDenseSolver) {
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (num_images <= kMaxNumImagesDirectSparseSolver) {
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {  // Indirect sparse (preconditioned CG) solver.
    solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

#ifdef OPENMP_ENABLED
  solver_options.num_threads =
       GetEffectiveNumThreads(solver_options.num_threads);
  solver_options.num_linear_solver_threads =
       GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif

  std::string error;
  CHECK(solver_options.IsValid(&error)) << error;

  ceres::Solve(solver_options, problem_.get(), &summary_);

  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (options_.print_summary) {
    PrintHeading2("Rig Bundle adjustment report");
    PrintSolverSummary(summary_);
  }

  TearDown(reconstruction, *camera_rigs);

  return true;
}

void RigBundleAdjuster::SetUp(Reconstruction* reconstruction,
                              std::vector<CameraRig>* camera_rigs,
                              ceres::LossFunction* loss_function) {
  ComputeCameraRigPoses(*reconstruction, *camera_rigs);

  for (const image_t image_id : config_.Images()) {
    AddImageToProblem(image_id, reconstruction, camera_rigs, loss_function);
  }
  for (const auto point3D_id : config_.VariablePoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }
  for (const auto point3D_id : config_.ConstantPoints()) {
    AddPointToProblem(point3D_id, reconstruction, loss_function);
  }

  ParameterizeCameras(reconstruction);
  ParameterizePoints(reconstruction);
  ParameterizeCameraRigs(reconstruction);
}

void RigBundleAdjuster::TearDown(Reconstruction* reconstruction,
                                 const std::vector<CameraRig>& camera_rigs) {
  for (const auto& elem : image_id_to_camera_rig_) {
    const auto image_id = elem.first;
    const auto& camera_rig = *elem.second;
    auto& image = reconstruction->Image(image_id);
    ConcatenatePoses(*image_id_to_rig_qvec_.at(image_id),
                     *image_id_to_rig_tvec_.at(image_id),
                     camera_rig.RelativeQvec(image.CameraId()),
                     camera_rig.RelativeTvec(image.CameraId()), &image.Qvec(),
                     &image.Tvec());
  }
}

void RigBundleAdjuster::AddImageToProblem(const image_t image_id,
                                          Reconstruction* reconstruction,
                                          std::vector<CameraRig>* camera_rigs,
                                          ceres::LossFunction* loss_function) {
  Image& image = reconstruction->Image(image_id);
  Camera& camera = reconstruction->Camera(image.CameraId());

  const bool constant_pose = config_.HasConstantPose(image_id);
  const bool constant_tvec = config_.HasConstantTvec(image_id);

  double* qvec_data = nullptr;
  double* tvec_data = nullptr;
  double* rig_qvec_data = nullptr;
  double* rig_tvec_data = nullptr;
  double* camera_params_data = camera.ParamsData();
  CameraRig* camera_rig = nullptr;
  Eigen::Matrix3x4d rig_proj_matrix = Eigen::Matrix3x4d::Zero();

  if (image_id_to_camera_rig_.count(image_id) > 0) {
    CHECK(!constant_pose)
        << "Images contained in a camera rig must not have constant pose";
    CHECK(!constant_tvec)
        << "Images contained in a camera rig must not have constant tvec";
    camera_rig = image_id_to_camera_rig_.at(image_id);
    rig_qvec_data = image_id_to_rig_qvec_.at(image_id)->data();
    rig_tvec_data = image_id_to_rig_tvec_.at(image_id)->data();
    qvec_data = camera_rig->RelativeQvec(image.CameraId()).data();
    tvec_data = camera_rig->RelativeTvec(image.CameraId()).data();

    // Concatenate the absolute pose of the rig and the relative pose the camera
    // within the rig to detect outlier observations.
    Eigen::Vector4d rig_concat_qvec;
    Eigen::Vector3d rig_concat_tvec;
    ConcatenatePoses(*image_id_to_rig_qvec_.at(image_id),
                     *image_id_to_rig_tvec_.at(image_id),
                     camera_rig->RelativeQvec(image.CameraId()),
                     camera_rig->RelativeTvec(image.CameraId()),
                     &rig_concat_qvec, &rig_concat_tvec);
    rig_proj_matrix = ComposeProjectionMatrix(rig_concat_qvec, rig_concat_tvec);
  } else {
    // CostFunction assumes unit quaternions.
    image.NormalizeQvec();
    qvec_data = image.Qvec().data();
    tvec_data = image.Tvec().data();
  }

  // Collect cameras for final parameterization.
  CHECK(image.HasCamera());
  camera_ids_.insert(image.CameraId());

  // The number of added observations for the current image.
  size_t num_observations = 0;

  // Add residuals to bundle adjustment problem.
  for (const Point2D& point2D : image.Points2D()) {
    if (!point2D.HasPoint3D()) {
      continue;
    }

    Point3D& point3D = reconstruction->Point3D(point2D.Point3DId());
    assert(point3D.Track().Length() > 1);

    if (camera_rig != nullptr &&
        (!HasPointPositiveDepth(rig_proj_matrix, point3D.XYZ()) ||
         CalculateReprojectionError(point2D.XY(), point3D.XYZ(),
                                    rig_proj_matrix,
                                    camera) > rig_options_.max_reproj_error)) {
      continue;
    }

    num_observations += 1;
    point3D_num_images_[point2D.Point3DId()] += 1;

    ceres::CostFunction* cost_function = nullptr;

    if (camera_rig == nullptr) {
      if (constant_pose) {
        switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                    \
  case CameraModel::kModelId:                                             \
    cost_function =                                                       \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(    \
            image.Qvec(), image.Tvec(), point2D.XY());                    \
    problem_->AddResidualBlock(cost_function, loss_function,              \
                               point3D.XYZ().data(), camera_params_data); \
    break;

          CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }
      } else {
        switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                   \
  case CameraModel::kModelId:                                            \
    cost_function =                                                      \
        BundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
    problem_->AddResidualBlock(cost_function, loss_function, qvec_data,  \
                               tvec_data, point3D.XYZ().data(),          \
                               camera_params_data);                      \
    break;

          CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
        }
      }
    } else {
      switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                      \
  case CameraModel::kModelId:                                               \
    cost_function =                                                         \
        RigBundleAdjustmentCostFunction<CameraModel>::Create(point2D.XY()); \
    problem_->AddResidualBlock(cost_function, loss_function, rig_qvec_data, \
                               rig_tvec_data, qvec_data, tvec_data,         \
                               point3D.XYZ().data(), camera_params_data);   \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
      }
    }
  }

  if (num_observations > 0) {
    parameterized_qvec_data_.insert(qvec_data);

    if (camera_rig != nullptr) {
      parameterized_qvec_data_.insert(rig_qvec_data);

      // Set the relative pose of the camera constant if relative pose
      // refinement is disabled or if it is the reference camera to avoid over-
      // parameterization of the camera pose.
      if (!rig_options_.refine_relative_poses ||
          image.CameraId() == camera_rig->RefCameraId()) {
        problem_->SetParameterBlockConstant(qvec_data);
        problem_->SetParameterBlockConstant(tvec_data);
      }
    }

    // Set pose parameterization.
    if (!constant_pose && constant_tvec) {
      const std::vector<int>& constant_tvec_idxs =
          config_.ConstantTvec(image_id);
      ceres::SubsetParameterization* tvec_parameterization =
          new ceres::SubsetParameterization(3, constant_tvec_idxs);
      problem_->SetParameterization(tvec_data, tvec_parameterization);
    }
  }
}

void RigBundleAdjuster::AddPointToProblem(const point3D_t point3D_id,
                                          Reconstruction* reconstruction,
                                          ceres::LossFunction* loss_function) {
  Point3D& point3D = reconstruction->Point3D(point3D_id);

  // Is 3D point already fully contained in the problem? I.e. its entire track
  // is contained in `variable_image_ids`, `constant_image_ids`,
  // `constant_x_image_ids`.
  if (point3D_num_images_[point3D_id] == point3D.Track().Length()) {
    return;
  }

  for (const auto& track_el : point3D.Track().Elements()) {
    // Skip observations that were already added in `AddImageToProblem`.
    if (config_.HasImage(track_el.image_id)) {
      continue;
    }

    point3D_num_images_[point3D_id] += 1;

    Image& image = reconstruction->Image(track_el.image_id);
    Camera& camera = reconstruction->Camera(image.CameraId());
    const Point2D& point2D = image.Point2D(track_el.point2D_idx);

    // We do not want to refine the camera of images that are not
    // part of `constant_image_ids_`, `constant_image_ids_`,
    // `constant_x_image_ids_`.
    if (camera_ids_.count(image.CameraId()) == 0) {
      camera_ids_.insert(image.CameraId());
      config_.SetConstantCamera(image.CameraId());
    }

    ceres::CostFunction* cost_function = nullptr;

    switch (camera.ModelId()) {
#define CAMERA_MODEL_CASE(CameraModel)                                     \
  case CameraModel::kModelId:                                              \
    cost_function =                                                        \
        BundleAdjustmentConstantPoseCostFunction<CameraModel>::Create(     \
            image.Qvec(), image.Tvec(), point2D.XY());                     \
    problem_->AddResidualBlock(cost_function, loss_function,               \
                               point3D.XYZ().data(), camera.ParamsData()); \
    break;

      CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }
  }
}

void RigBundleAdjuster::ComputeCameraRigPoses(
    const Reconstruction& reconstruction,
    const std::vector<CameraRig>& camera_rigs) {
  camera_rig_qvecs_.reserve(camera_rigs.size());
  camera_rig_tvecs_.reserve(camera_rigs.size());
  for (const auto& camera_rig : camera_rigs) {
    camera_rig_qvecs_.emplace_back();
    camera_rig_tvecs_.emplace_back();
    auto& rig_qvecs = camera_rig_qvecs_.back();
    auto& rig_tvecs = camera_rig_tvecs_.back();
    rig_qvecs.resize(camera_rig.NumSnapshots());
    rig_tvecs.resize(camera_rig.NumSnapshots());
    for (size_t snapshot_idx = 0; snapshot_idx < camera_rig.NumSnapshots();
         ++snapshot_idx) {
      camera_rig.ComputeAbsolutePose(snapshot_idx, reconstruction,
                                     &rig_qvecs[snapshot_idx],
                                     &rig_tvecs[snapshot_idx]);
      for (const auto image_id : camera_rig.Snapshots()[snapshot_idx]) {
        image_id_to_rig_qvec_.emplace(image_id, &rig_qvecs[snapshot_idx]);
        image_id_to_rig_tvec_.emplace(image_id, &rig_tvecs[snapshot_idx]);
      }
    }
  }
}

void RigBundleAdjuster::ParameterizeCameraRigs(Reconstruction* reconstruction) {
  for (double* qvec_data : parameterized_qvec_data_) {
    ceres::LocalParameterization* quaternion_parameterization =
        new ceres::QuaternionParameterization;
    problem_->SetParameterization(qvec_data, quaternion_parameterization);
  }
}

void PrintSolverSummary(const ceres::Solver::Summary& summary) {
  std::cout << std::right << std::setw(16) << "Residuals : ";
  std::cout << std::left << summary.num_residuals_reduced << std::endl;

  std::cout << std::right << std::setw(16) << "Parameters : ";
  std::cout << std::left << summary.num_effective_parameters_reduced
            << std::endl;

  std::cout << std::right << std::setw(16) << "Iterations : ";
  std::cout << std::left
            << summary.num_successful_steps + summary.num_unsuccessful_steps
            << std::endl;

  std::cout << std::right << std::setw(16) << "Time : ";
  std::cout << std::left << summary.total_time_in_seconds << " [s]"
            << std::endl;

  std::cout << std::right << std::setw(16) << "Initial cost : ";
  std::cout << std::right << std::setprecision(6)
            << std::sqrt(summary.initial_cost / summary.num_residuals_reduced)
            << " [px]" << std::endl;

  std::cout << std::right << std::setw(16) << "Final cost : ";
  std::cout << std::right << std::setprecision(6)
            << std::sqrt(summary.final_cost / summary.num_residuals_reduced)
            << " [px]" << std::endl;

  std::cout << std::right << std::setw(16) << "Termination : ";

  std::string termination = "";

  switch (summary.termination_type) {
    case ceres::CONVERGENCE:
      termination = "Convergence";
      break;
    case ceres::NO_CONVERGENCE:
      termination = "No convergence";
      break;
    case ceres::FAILURE:
      termination = "Failure";
      break;
    case ceres::USER_SUCCESS:
      termination = "User success";
      break;
    case ceres::USER_FAILURE:
      termination = "User failure";
      break;
    default:
      termination = "Unknown";
      break;
  }

  std::cout << std::right << termination << std::endl;
  std::cout << std::endl;
}

void PrintErrorStatistics(const Reconstruction* reconstruction, double max_fitting_error) {
  std::vector<double> errors;
  double max_error = DBL_MIN;
  int beyond_limit = 0;
  for (const auto &view_it : reconstruction->Images())
  {
    const Image *prior = &view_it.second;
    if (prior != nullptr && prior->HasTvecPrior() && prior->TvecPrior().sum() != 0 && prior->HasCamera() && reconstruction->IsImageRegistered(view_it.first))
    {
      double error = (prior->ProjectionCenter() - prior->TvecPrior()).norm();
      max_error = error > max_error ? error : max_error;
      errors.push_back(error);
      if (error > max_fitting_error) beyond_limit++;
    }
  }
  
  cout << "Reconstruction Fitting error statistics:" << endl
      << " #poses: " << errors.size() << endl;
  
  if (!errors.empty()) {
    cout<< " #mean: " << Mean(errors) << endl
        << " #median: " << Median(errors) << endl
        << " #max: " << max_error << endl
        << " #beyond " << max_fitting_error << ": " << beyond_limit << endl
        << endl;
  }
}
}  // namespace colmap

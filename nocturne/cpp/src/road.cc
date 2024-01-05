// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "road.h"

#include "geometry/vector_2d.h"
#include "utils/sf_utils.h"

namespace nocturne {

sf::Color RoadTypeColor(const RoadType& road_type) {
  switch (road_type) {
    case RoadType::kLane: {
      return sf::Color::Yellow;
    }
    case RoadType::kRoadLine: {
      return sf::Color::Blue;
    }
    case RoadType::kRoadEdge: {
      return sf::Color::Green;
    }
    case RoadType::kStopSign: {
      return sf::Color::Red;
    }
    case RoadType::kCrosswalk: {
      return sf::Color::Magenta;
    }
    case RoadType::kSpeedBump: {
      return sf::Color::Cyan;
    }
    default: {
      return sf::Color::Transparent;
    }
  };
}

sf::Color RoadLine::Color() const { return RoadTypeColor(road_type_); }

void RoadLine::draw(sf::RenderTarget& target, sf::RenderStates states) const {
  target.draw(graphic_points_.data(), graphic_points_.size(), sf::LineStrip,
              states);
}

void RoadLine::InitRoadPoints() {
  const int64_t num_segments = geometry_points_.size() - 1;
  const int64_t num_sampled_points =
      (num_segments + sample_every_n_ - 1) / sample_every_n_ + 1;
  road_points_.reserve(num_sampled_points);
  if (road_type_ == RoadType::kLane || road_type_ == RoadType::kRoadEdge || road_type_ == RoadType::kRoadLine) {
    std::vector<bool> skip(num_sampled_points, false); // This list tracks the points that are skipped
    int64_t j = 0;
    bool skipChanged = true; // This is used to check if the skip list has changed in the last iteration
    while (skipChanged) // This loop runs O(N^2) in worst case, but it is very fast in practice probably O(NlogN)
    {
      skipChanged = false; // Reset the skipChanged flag
      j = 0;
      while (j < num_sampled_points - 1)
      {
          int64_t j_1 = j + 1; // j_1 is the next point that is not skipped
          while (j_1 < num_sampled_points - 1 && skip[j_1]) {
              j_1++; // Keep incrementing j_1 until we find a point that is not skipped
          }
          if(j_1 >= num_sampled_points - 1)
              break;
          int64_t j_2 = j_1 + 1;
          while (j_2 < num_sampled_points && skip[j_2]) {
              j_2++; // Keep incrementing j_2 until we find a point that is not skipped
          }
          if(j_2 >= num_sampled_points)
              break;
          auto point1 = geometry_points_[j * sample_every_n_];
          auto point2 = geometry_points_[j_1 * sample_every_n_];
          auto point3 = geometry_points_[j_2 * sample_every_n_];
          float_t area = 0.5 * std::abs((point1.x() - point3.x()) * (point2.y() - point1.y()) - (point1.x() - point2.x()) * (point3.y() - point1.y()));
          if (area < reducing_threshold_) { // If the area is less than the threshold, then we skip the middle point
              skip[j_1] = true; // Mark the middle point as skipped
              j = j_2;  // Skip the middle point and start from the next point
              skipChanged = true; // Set the skipChanged flag to true
          }
          else
          {
              j = j_1; // If the area is greater than the threshold, then we don't skip the middle point and start from the next point
          }
      }
    }
  
    // Create the road lines
    j = 0;
    skip[0] = false;
    skip[num_sampled_points - 1] = false;
    std::vector<geometry::Vector2D> new_geometry_points; // This list stores the points that are not skipped
    while (j < num_sampled_points)
    {
      if (!skip[j])
      {
        new_geometry_points.push_back(geometry_points_[j * sample_every_n_]); // Add the point to the list if it is not skipped
      }
      j++;
    }
    for(int64_t i = 0; i < new_geometry_points.size() - 1; i++)
    {
      road_points_.emplace_back(new_geometry_points[i], new_geometry_points[i + 1], road_type_); // Create the road lines
    }
  
    road_points_.emplace_back(geometry_points_[num_sampled_points - 2], geometry_points_.back(), road_type_); // Create the last road line
    road_points_.emplace_back(geometry_points_.back(), geometry_points_.back(), road_type_); // Use itself as neighbor for the last point.

    // This is the same logic as before but more efficient without creating a new vector
    // But I am using the above logic for now to make it simple to debug. 
    // The missing edges bug is a problem in both logics

    // while( j < num_sampled_points)
    // {
    //   int64_t j_1 = j + 1;
    //   while (j_1 < num_sampled_points && skip[j_1]) {
    //       j_1++;
    //   }
    //   if (j_1 == num_sampled_points) {

    //     if (j != num_sampled_points - 1){
    //       // std::cout<<"making an edge from "<<j<<" to "<<num_sampled_points - 1<<std::endl;
    //       road_points_.emplace_back(geometry_points_[j * sample_every_n_],
    //                                 geometry_points_.back(), road_type_);
    //     }
    //     break;
    //   }
    //   std::cout<<"making an edge from "<<j<<" to "<<j_1<<std::endl;
    //   road_points_.emplace_back(geometry_points_[j * sample_every_n_],
    //                                       geometry_points_[(j_1) * sample_every_n_],
    //                                       road_type_);
    //   j = j_1;
    // }

  }
   else {
    for (int64_t i = 0; i < num_sampled_points - 2; ++i) {
      road_points_.emplace_back(geometry_points_[i * sample_every_n_],
                                geometry_points_[(i + 1) * sample_every_n_],
                                road_type_);
    }
    const int64_t p = (num_sampled_points - 2) * sample_every_n_;
    road_points_.emplace_back(geometry_points_[p], geometry_points_.back(),
                              road_type_);
    // Use itself as neighbor for the last point.
    road_points_.emplace_back(geometry_points_.back(), geometry_points_.back(),
                              road_type_);
  }
}

void RoadLine::InitRoadLineGraphics() {
  // const int64_t n = geometry_points_.size();
  // graphic_points_.reserve(n);
  // for (const geometry::Vector2D& p : geometry_points_) {
  //   graphic_points_.emplace_back(sf::Vertex(utils::ToVector2f(p), Color()));
  // }
  const int64_t n = road_points_.size();
  graphic_points_.reserve(n);
  for (const RoadPoint& p : road_points_) {
    graphic_points_.emplace_back(sf::Vertex(utils::ToVector2f(p.position()), Color()));
  }

}

}  // namespace nocturne

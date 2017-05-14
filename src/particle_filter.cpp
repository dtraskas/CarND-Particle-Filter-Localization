/*
 * particle_filter.cpp
 *
 *  Created on: May 14, 2017
 *  Author: Dimitrios Traskas
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.

	num_particles = 100;
	// Gaussian noise distributions
	normal_distribution<double> dist_x(0, std[0]);
	normal_distribution<double> dist_y(0, std[1]);
	normal_distribution<double> dist_theta(0, std[2]);

	// Initialize particles with random particle position and noise, and weight of 1
	default_random_engine rnd;
	for (int i = 0; i <= num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = x + dist_x(rnd);
		particle.y = y + dist_y(rnd);
		particle.theta = theta + dist_theta(rnd);
		particle.weight = 1.0;

		weights.push_back(particle.weight);
		particles.push_back(particle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> n_x(0, std_pos[0]);
	normal_distribution<double> n_y(0, std_pos[1]);
	normal_distribution<double> n_theta(0, std_pos[2]);

	default_random_engine rnd;
	for (unsigned int i=0; i < particles.size(); ++i) {
		double x_pred;
		double y_pred;
		double theta_pred;

		// Check the value of yaw rate if close to zero
		if (yaw_rate < 0.001) {
			x_pred = particles[i].x + velocity * cos(particles[i].theta) * delta_t;
			y_pred = particles[i].y + velocity * sin(particles[i].theta) * delta_t;
			theta_pred = particles[i].theta + yaw_rate * delta_t;
		} else {
			x_pred = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			y_pred = particles[i].y + (velocity / yaw_rate) * (-cos(particles[i].theta + yaw_rate * delta_t) + cos(particles[i].theta));
			theta_pred = particles[i].theta + yaw_rate * delta_t;
		}

		// Update particle with noise added
		particles[i].x = x_pred + n_x(rnd);
		particles[i].y = y_pred + n_y(rnd);
		particles[i].theta = theta_pred + n_theta(rnd);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned i = 0; i < observations.size(); ++i) {
		double dist_current = 1e6;
		int nearest_landmark = -1;

		for (unsigned k = 0; k < predicted.size(); ++k) {
			double distance = dist(observations[i].x, observations[i].y, predicted[k].x, predicted[k].y);
			if (distance < dist_current) {
				dist_current = distance;
				nearest_landmark = k;
			}
		}
		// assign the closest landmark to the obeservation
		observations[i].id = predicted[nearest_landmark].id;
	}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	weights.clear();
	for (unsigned i = 0; i <= particles.size(); i++) {
		// Transform observation coordinates from vehicle to map
		std::vector<LandmarkObs> observation_coords;
		for (unsigned int k = 0; k < observations.size(); ++k) {
			if (dist(observations[k].x, observations[k].y, 0, 0) <= sensor_range) {
				LandmarkObs obs;
				obs.x = particles[i].x + observations[k].x * cos(particles[i].theta) - observations[k].y * sin(particles[i].theta);
				obs.y = particles[i].y + observations[k].x * sin(particles[i].theta) + observations[k].y * cos(particles[i].theta);
				obs.id = -1;
				observation_coords.push_back(obs);
			}
		}

		// store the nearest landmarks
		std::vector<LandmarkObs> nearest_landmarks;
		for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
			if (dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f) <= sensor_range) {
				LandmarkObs obs;
				obs.x = map_landmarks.landmark_list[k].x_f;
				obs.y = map_landmarks.landmark_list[k].y_f;
				obs.id = map_landmarks.landmark_list[k].id_i;
				nearest_landmarks.push_back(obs);
			}
		}

		// Get the nearest landmark for each observaton
		dataAssociation(nearest_landmarks, observation_coords);

		// Calculate new weights by calculating Gaussian probabilities
		double weight = 1;
		for (unsigned int j = 0; j <= nearest_landmarks.size(); ++j) {
			double dist_min = 1e6;
			int min_k = -1;
			for (unsigned int k = 0; k < observation_coords.size(); ++k) {
				if (observation_coords[k].id == nearest_landmarks[j].id) {
					double euclidean_distance = dist(nearest_landmarks[j].x, nearest_landmarks[j].y, observation_coords[k].x, observation_coords[k].y);
					if (euclidean_distance < dist_min) {
						dist_min = euclidean_distance;
						min_k = k;
					}
				}
			}
			
			if (min_k > 0) {
				double x = observation_coords[min_k].x;
				double y = observation_coords[min_k].y;
				
				weight *= exp(-((x - nearest_landmarks[j].x) * (x - nearest_landmarks[j].x) / (2 * std_landmark[0] * std_landmark[0]) + (y - nearest_landmarks[j].y) * (y - nearest_landmarks[j].y) / (2 * std_landmark[1] * std_landmark[1]))) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
			}
		}
		weights.push_back(weight);
		particles[i].weight = weight;
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine rnd;
	discrete_distribution<int> dist(weights.begin(), weights.end());
	std::vector<Particle> new_particles;
	for (int i = 0; i <= num_particles; i++) {
		new_particles.push_back(particles[dist(rnd)]);
	}
	particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
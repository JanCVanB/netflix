#include <stdio.h>

int c_train_epoch(int *train_points, int num_points, float *users, float *user_offsets,
        int num_users, float *movies, float* movie_averages, int num_movies,
        float learn_rate, int num_features, float k_factor)
{

	int p, f;
	float prediction;
	float *user_features_cursor, *movie_features_cursor;
	float error, user_change, movie_change;
	int *train_cursor = train_points;
	int user_id, movie_id, time, rating;

	for(p = 0; p < num_points; p++) {
	    // get user id's locally
        //user_id = train_points[p*4];
        //movie_id = train_points[p*4+1];
        //rating = train_points[p*4+3];
		user_id    = *(train_cursor++);
		movie_id   = *(train_cursor++);
		time       = *(train_cursor++);
		rating     = *(train_cursor++);
        // Calculate the prediction error:
        // start prediction at baseline:
        prediction = movie_averages[movie_id] + user_offsets[user_id];
        // then: add features dot product to prediction
        user_features_cursor = users + user_id * num_features;
        movie_features_cursor = movies + movie_id * num_features;
        for (f = 0; f < num_features; f++) {
            prediction += user_features_cursor[f] * movie_features_cursor[f];
            if (prediction > 5) {
                prediction = 5.0;
            } else if (prediction < 1) {
                prediction = 1.0;
            }
        }

        // Calculate error:
        error = ((float) rating) - prediction;

        // Update the features
        // reset feature cursors
        for (f = 0; f < num_features; f++) {
            user_change = learn_rate * (error * movie_features_cursor[f] 
                - k_factor * user_features_cursor[f]);
            movie_features_cursor[f] += learn_rate * (error * user_features_cursor[f] 
                - k_factor * movie_features_cursor[f]);
            user_features_cursor[f] += user_change;
        }

	}
    return 0;
}



int c_update_feature(int *train_points, int num_points, float *users, 
        int num_users, float *movies, int num_movies, float learn_rate, 
        int feature, int num_features){

	int p, f;
	float prediction;
	float *user_features, *movie_features;
	float error, user_change, movie_change;
	int *user,*movie,*time,*rating;
	int *train_cursor = train_points;
	
	for(p = 0; p < num_points; p++){
		/* Get current variables   */
		user   = train_cursor++;
		movie  = train_cursor++;
		time   = train_cursor++;
		rating = train_cursor++; 
		
		/* Calculate prediction error */
		/* TODO: Use residuals to avoid repeat dot product calculations */
		user_features  = users + (*user * num_features);
		movie_features = movies + (*movie * num_features);
        prediction = 0;
		for(f = 0; f < num_features; f++){
			prediction += user_features[f] * movie_features[f];
		}
		
		error = *rating - prediction;
		/* Update user and movie */
		user_change  = learn_rate * error * movie_features[feature];
		movie_change = learn_rate * error * user_features[feature];
		
		user_features[feature]  += user_change;
		movie_features[feature] += movie_change;
		/* */
	}
    return 0;
}
int c_update_feature_with_pointers(int *train_points, int num_points, float *users,
        int num_users, float *movies, int num_movies, float learn_rate,
        int feature, int num_features){

	int p, f;
	float prediction;
	float *user_features, *movie_features;
	float *user_features_cursor, *movie_features_cursor;
	float error, user_change, movie_change;
	int *user,*movie,*time,*rating;
	int *train_cursor = train_points;

	for(p = 0; p < num_points; p++){
		/* Get current variables   */
		user   = train_cursor++;
		movie  = train_cursor++;
		time   = train_cursor++;
		rating = train_cursor++;

		/* Calculate prediction error */
		/* TODO: Use residuals to avoid repeat dot product calculations */
		user_features  = users + (*user * num_features);
		movie_features = movies + (*movie * num_features);
		user_features_cursor  = user_features;
		movie_features_cursor = movie_features;
        prediction = 0;
		for(f = 0; f < num_features; f++){
			prediction += *user_features_cursor * *movie_features_cursor;
			user_features_cursor++;
			movie_features_cursor++;
		}
		error = *rating - prediction;

		user_features_cursor  = user_features[feature];
		movie_features_cursor = movie_features[feature];
		/* Update user and movie */
		user_change  = learn_rate * error * *movie_features_cursor;
		movie_change = learn_rate * error * *user_features_cursor;

		*user_features_cursor  += user_change;
		*movie_features_cursor += movie_change;
		/* */
	}
    return 0;
}

int c_update_feature_with_residuals(int *train_points, int num_points, float *users,
        int num_users, float *movies, int num_movies, float *residuals, float learn_rate,
        int feature, int num_features){

	int p, f;
	float prediction;
	float *user_features, *movie_features;
	float *user_features_cursor, *movie_features_cursor;
	float error, user_change, movie_change;
	int *user,*movie,*time,*rating;
	int *train_cursor = train_points;

	for(p = 0; p < num_points; p++){
		/* Get current variables   */
		user   = train_cursor++;
		movie  = train_cursor++;
		time   = train_cursor++;
		rating = train_cursor++;

		/* Calculate prediction error */
		/* TODO: Use residuals to avoid repeat dot product calculations */
		user_features  = users + (*user * num_features);
		movie_features = movies + (*movie * num_features);
		user_features_cursor  = user_features;
		movie_features_cursor = movie_features;
        prediction = 0;
		for(f = 0; f < num_features; f++){
			prediction += *user_features_cursor * *movie_features_cursor;
			user_features_cursor++;
			movie_features_cursor++;
		}
		error = *rating - prediction;

		user_features_cursor  = user_features[feature];
		movie_features_cursor = movie_features[feature];
		/* Update user and movie */
		user_change  = learn_rate * error * *movie_features_cursor;
		movie_change = learn_rate * error * *user_features_cursor;

		*user_features_cursor  += user_change;
		*movie_features_cursor += movie_change;
		/* */
	}
    return 0;
}
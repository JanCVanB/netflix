int c_update_feature(int *train_points, int num_points, float *users, 
        int num_users, float *movies, int num_movies, float learn_rate, 
        int feature, int num_features){
	
	int i, j;
	float temp_prediction;
	int *temp_user_val, *temp_movie_val;
	float error, user_change, movie_change;
	int *user,*movie,*time,*rating;
	int *train_cursor = train_points;
	
	for(i = 0; i < num_points; i++){
		/* Get current variables   */
		user   = train_cursor++;
		movie  = train_cursor++;
		time   = train_cursor++;
		rating = train_cursor++; 
		
		/* Calculate prediction error */
		/* TODO: Use residuals to avoid repeat dot product calculations */
		temp_user_val  = users  + (*user * num_features);
		temp_movie_val = movies + (*movie * num_features);
		for(j = 0; j < num_features; j++){
			temp_prediction += temp_user_val[j] * temp_movie_val[j];
			
		}
		
		error = *rating - temp_prediction;
		/* Update user and movie */
		user_change  = learn_rate * error * movies[(*movie * num_features) + feature];
		movie_change = learn_rate * error * users[(*user  * num_features) + feature];
		
		users[(*movie * num_features) + feature]   += user_change;
		movies[(*user  * num_features) + feature]  += movie_change;
		/* */
	}
}

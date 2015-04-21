int py_c_update_feature(int *train_points, int num_points, float *users, 
	int num_users, float *movies, int num_movies, float learn_rate, int feature){
	
	int i, j;
	float temp_prediction;
	float error, user_change, movie_change;
	int *user,*movie,*time,*rating;
	int *train_cursor = train_points;
	
	for(i = 0; i < num_points; i++){
		/* Get current variables   */
		user   = (&train_points++;
		movie  = &train_points++;
		time   = &train_points++;
		rating = &train_points++; 
		
		/* Calculate prediction error */
		/* TODO: Use residuals to avoid repeat dot product calculations */
		for(j = 0; j < num_features; j++){
			temp_prediction += users[*user][j] * movies[j][*movie];
			
		}
		
		error = rating - temp_prediction;
		/* Update user and movie */
		user_change = learn_rate * error * movies[feature][*movie];
		movie_change = learn_rate * error * users[*user][feature];
		
		users[*user][feature]   += user_change;
		movies[feature][*movie] += movie_change;
		/* */
	}
}
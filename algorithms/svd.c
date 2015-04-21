int py_c_update_feature(int *train_points, int num_points, float *users, 
	int num_users, float *movies, int num_movies, float learn_rate, int feature){
	int i;
	float error, user_change, movie_change;
	int *user,*movie,*time,*rating;
	int *train_cursor = train_points;
	
	for(i = 0; i < num_points; i++){
		/* Get current variables   */
		user   = train_points++;
		movie  = train_points++;
		time   = train_points++;
		rating = train_points++; 
		/* Calculate prediction error */
		error = rating - calculate_prediction(user, movie);
		/* Update user and movie */
		user_change = learn_rate * error * movies[feature][movie];
		movie_change = learn_rate * error * users[user][feature];
		/* */
	}
}
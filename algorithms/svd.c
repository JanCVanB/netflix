int static py_c_update_feature(int *train_points, int num_points, float *u, 
	int num_users, float *v, int num_movies){
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
		
		/* */
	}
}
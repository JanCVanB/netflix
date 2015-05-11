#include <stdio.h>

int c_update_feature(int *train_points, int num_points, float *users,
        int num_users, float *movies, int num_movies, float *residuals, float learn_rate,
        int feature, int num_features){

	int p, f;
	float prediction, feature_product;
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
		user_features  = users + (*user * num_features);
	    movie_features = movies + (*movie * num_features);
       	user_features_cursor  = user_features;
      	movie_features_cursor = movie_features;
        prediction = 0;

		feature_product = user_features[feature]*movie_features[feature];
		if(feature == 0){
       		 for(f = 0; f < num_features; f++){
	                prediction += *user_features_cursor * *movie_features_cursor;
        	        user_features_cursor++;
               		movie_features_cursor++;
                  }
                 // if(p ==  0 || p == num_points-1){
               	 //	printf("Feature number %d\n", feature);
              	  //}

		}else{
		    prediction  = residuals[p];
		    prediction += user_features[feature-1]*movie_features[feature-1];
            //feature_product = user_features[feature]*movie_features[feature];
		    prediction += feature_product;
		    //if(p == 0 || p == num_points-1){
                    //    printf("First prediction with residual:  %f\n", prediction);
		    //	printf("Last/first previous user features: %f, movie features: %f\n",
		    //		user_features[feature-1],movie_features[feature-1]);
		    //	}
		}
		
		error = ((float) *rating) - prediction;

		user_features_cursor  = &user_features[feature];
		movie_features_cursor = &movie_features[feature];

		/* Update user and movie */
		user_change  = learn_rate * error * *movie_features_cursor;
		movie_change = learn_rate * error * *user_features_cursor;


		*user_features_cursor  += user_change;
		*movie_features_cursor += movie_change;

		user_features_cursor++; //next feature
		movie_features_cursor++;

      	/* mmm save the residual */
		if(feature < num_features - 1){
		    //residuals[p] = prediction - *user_features_cursor * *movie_features_cursor;
		    residuals[p] = prediction - feature_product;
	        residuals[p] -= (*(user_features_cursor) * *(movie_features_cursor));
		}
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

		user_features_cursor  = &user_features[feature];
		movie_features_cursor = &movie_features[feature];
		/* Update user and movie */
		user_change  = learn_rate * error * *movie_features_cursor;
		movie_change = learn_rate * error * *user_features_cursor;

		*user_features_cursor  += user_change;
		*movie_features_cursor += movie_change;
		/* */
		if(p == 0)
		printf("First prediction %f\n", prediction);
	}
    return 0;
}

int c_update_feature_with_residuals(int *train_points, int num_points, float *users,
        int num_users, float *movies, int num_movies, float *residuals, float learn_rate,
        int feature, int num_features){

	int p, f;
	float prediction, feature_product;
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
		user_features  = users + (*user * num_features);
	    movie_features = movies + (*movie * num_features);
       	user_features_cursor  = user_features;
      	movie_features_cursor = movie_features;
        prediction = 0;

        feature_product = user_features[feature]*movie_features[feature];
		if(feature == 0){
       		 for(f = 0; f < num_features; f++){
	                prediction += *user_features_cursor * *movie_features_cursor;
        	        user_features_cursor++;
               		movie_features_cursor++;
                  }
                 // if(p ==  0 || p == num_points-1){
               	 //	printf("Feature number %d\n", feature); 
              	  //}         
  	 
		}else{	
		    prediction  = residuals[p];
		    prediction += user_features[feature-1]*movie_features[feature-1];
            //feature_product = user_features[feature]*movie_features[feature];
		    prediction += feature_product;
		    //if(p == 0 || p == num_points-1){
                    //    printf("First prediction with residual:  %f\n", prediction);
		    //	printf("Last/first previous user features: %f, movie features: %f\n",
		    //		user_features[feature-1],movie_features[feature-1]);
		    //	}
		}
		error = ((float) *rating) - prediction;
		//if(p==0 || p==num_points-1){
		// printf("First error:  %f\n", error);
		//}
		user_features_cursor  = &user_features[feature];
		movie_features_cursor = &movie_features[feature];

		/* Update user and movie */
		user_change  = learn_rate * error * *movie_features_cursor;
		movie_change = learn_rate * error * *user_features_cursor;
		//if(p== num_points-1){
		//  printf("last element user_feature_val: %f, movie_feature_val: %f, and their producr: %f\n", 
		//		*user_features_cursor, *movie_features_cursor,
		//		*user_features_cursor * *movie_features_cursor);	
		//  printf("last element user change: %f and movie change: %f\n", user_change, movie_change);
		//}

		*user_features_cursor  += user_change;
		*movie_features_cursor += movie_change;
		//if(p==num_points-1){
		// printf("last element updated user_feature: %f, movie_feature: %f\n", 
		//	*user_features_cursor, *movie_features_cursor);
		//}
		user_features_cursor++; //next feature
		movie_features_cursor++;

      	/* mmm save the residual */
		if(feature < num_features - 1){
		    //residuals[p] = prediction - *user_features_cursor * *movie_features_cursor;
		    residuals[p] = prediction - feature_product;
	        residuals[p] -= (*(user_features_cursor) * *(movie_features_cursor));
		}
		//if(p==num_points-1){
		// printf("last element residual: %f, and prediction: %f\n", residuals[p], prediction);
		//}
	}
	//printf("First residual %f\n", residuals[0]);
        //printf("Second residual %f\n", residuals[1]);
	//printf("Second to last residual %f\n", residuals[num_points - 2]);
	//printf("Last residual %f\n", residuals[num_points-1]);
    return 0;
}

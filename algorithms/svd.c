#include <stdio.h>

int c_update_feature(int *train_points, int num_points, float *users, float *user_offsets,
        int num_users, float *movies, float* movie_averages, int num_movies, float *residuals,
        float learn_rate, int feature, int num_features)
{

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
        prediction = user_offsets[*user] + movie_averages[*movie];

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



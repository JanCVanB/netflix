#include <stdio.h>
#include <math.h>

float compute_implicit_preferences_sum(int num_points, float *implicit_preferences_cursor);

int c_run_svd_plus_epoch(
        int *train_points, int num_points,
        float *users, float *user_offsets, int *user_rating_count, int num_users,
        float *movies, float* movie_averages, int *movie_rating_count, int num_movies,
        float *similarity_matrix_rated, int num_neighbors,   // assume similarity matrix is sorted 2dimensional
        int *nearest_neighbors_matrix,
        float *implicit_preferences,
        float *explicit_feedback,
        float *implicit_feedback,
        int num_features,
        float offset_learn_rate, float feature_learn_rate, float feedback_learn_rate,
        float offset_k_factor, float feature_k_factor, float feedback_k_factor)
{

    // r = user_offset + movie_average + movies * [users + (num_ratings)*\Sum_j{w_j}
    // \sum over(j in R(u)[ (r_uj - b_uj)* w_ij ]  - going through all movies rated by user (u). b_uj is user offset for movie

    // From Koren paper on SVD++, the equivalent variables used
    // lambda1 = offset_learn_rate, lambda2 = feature_learn_rate, lambda3 = feedback_learn_rate,
    // lambda6 = offset_k_factor, lambda7 = feature_k_factor, lambda8 = feedback_k_factor
    // b_u = user_offset                                                        [num_users, 1]
    // b_i = movie_average (offset)                                             [num_movies, 1]
    // q_i = *movies                                                            [num_movies, num_features]
    // p_u = *users                                                             [num_users, num_features]
    // e_ui = rating[u][i] - predicted[u][i]                                    [num_points, 1]
    // y_j  = *user_implicit_preferences                                        [num_movies, 1]
    // w_ij = *explicit_feedback                                                [num_movies, num_movies]
    // c_ij = *implicit_feedback                                                [num_movies, num_movies]
    // N(u) = movies not rated by user (u)                                      [num_users, .....]
    // R^k(u) = k movies rated by user (u) that are similar to i                [num_users, .....]
    // N^k(u) = k movies NOT rated by user (u) that are most similar to i       [num_users, .....]


   //For (num_points) loop:
    // user_offset  += offset_learn_rate * (error - offset_k_factor * user_offset)
    // movie_offset += offset_learn_rate * (error - offset_k_factor * movie_offset)
    // For j in (num_movies_not_rated by u) loop
    //      user_implicit_sum += user_implicit_preferences[j]
    // user_implicit_sum *= sqrt(|N(u)|)
    // movies[i] += feature_learn_rate * (error * (users[u] + user_implicit_sum) - feature_k_factor*movies[i])
    // users[u]  += feature_learn_rate * (error * movies[i] - feature_k_factor * users[u])

    // For j in (num_movies_not_rated by u) loop
    //      user_implicit_preferences[j] += feature_learn_rate * (error * sqrt(|N(u)|) * movies[j] - feature_k_factor*user_implicit_preferences[j])

    // num_rated_and_similar = |R^k(u)|
    // For j in (num_rated_and_similar) loop
    //      explicit_feedback[i][j] += feedback_learn_rate * (sqrt(num_rated_and_similar) * error
    //          * (rating - (user_offset[u]+movie_average[i]) - (feedback_k_factor * explicit_feedback[i][j])

    // num_not_rated_and_similar = |N^k(u)|
    // For j in (num_not_rated_and_similar) loop
    //      implicit_feedback[i][j] += feedback_learn_rate * (sqrt(num_not_rated_and_similar) * error
    //          - (feedback_k_factor * explicit_feedback[i][j])


	int p, f, j, code_index;                                        /* loop indices */

	float prediction, feature_product;                      /* prediction variables */
	float implicit_preferences_sum, user_implicit_preferences_sum;
	float user_implicit_preferences_offset;
	float explicit_feedback_sum, implicit_feedback_sum;
    int num_ratings;
    float intermediate;
    float scalar_implicit_preference, scalar_implicit_feedback, scalar_explicit_feedback;
    float num_explicit_neighbors, num_implicit_neighbors;

	float *user_features, *movie_features;                  /* ptr to start of user/movie features */
	float *user_features_cursor, *movie_features_cursor;    /* cursor to run thru features */
	int *train_cursor = train_points;                       /* cursor for training list */
	float *implicit_preferences_cursor = implicit_preferences;

	float error, user_change, movie_change;                 /* weight updating variables */
	int *user,*movie,*time,*rating;                         /* actual test point values */
    int *current_user;                                      /* pointer to the head of a user's ratings */
    int similar_movie;
    int binary_encoding;



    for(p = 0; p < num_points; p++){
        implicit_preferences_sum = compute_implicit_preferences_sum(num_points, implicit_preferences_cursor);

    	/* Get current variables   */
		user   = train_cursor++;
		movie  = train_cursor++;
		time   = train_cursor++;
		rating = train_cursor++;

        /********************** Sum( N(u) of y_j ) *****************************/
        /********************** Implicit Preferences *****************************/
        if(*current_user != *user){          // are we training on a new user?
            current_user = user;           // if so, register the new user
            user_implicit_preferences_sum = 0;
            num_ratings = 0;
            train_cursor -= 4;  // get back to current user
            while(*train_cursor == *user){  // and calculate the implicit sum and count
                num_ratings++;  // count number of ratings by user
                train_cursor++; // get the movie being rated
                user_implicit_preferences_sum += implicit_preferences[*train_cursor];
                train_cursor += 3;
            }
            train_cursor -= (num_ratings-1)*4; // get back to the user index of next rating
        }

        /* TODO: change the scalar multiple to square root */
        scalar_implicit_preference = num_movies - user_rating_count[*user];
        scalar_implicit_preference = sqrt(scalar_implicit_preference);
        user_implicit_preferences_offset = (implicit_preferences_sum - user_implicit_preferences_sum)
                                               * (num_movies - user_rating_count[*user]);

        /********************** Sum( R^k( (r_uj - b_uj) * w_ij ) ) *****************************/
        /**********************  Sum for N^k(u) { c_ij } ***************************************/
        // sum of all the ratings by user (u) within the neighbor (up to k)
        explicit_feedback_sum = 0;
        num_explicit_neighbors = 0;
        implicit_feedback_sum = 0;
        num_implicit_neighbors = 0;
        code_index = 0;
        for(j = 0; j < num_neighbors; j++){
            if(j % 32 == 0){
                binary_encoding = nearest_neighbors_matrix[p*10 + code_index];
                code_index++;
            }
            if(binary_encoding & 0x8000 == 1){ /* Add to R^k(u) if rated, N^k(u) if not rated */
                similar_movie = similarity_matrix_rated[(*movie * num_neighbors) + j]; // get next most similar movie
                explicit_feedback_sum += (user_rating_for_similar[(*current_user  + j]
                    - (user_offsets[*current_user] + movie_averages[similar_movie]))
                    * explicit_feedback[*current_user][similar_movie];
                num_explicit_neighbors++:
            }else{
                similar_movie = similarity_matrix_rated[*movie][j];
                implicit_feedback_sum += implicit_feedback[*movie][similar_movie];
                num_implicit_neighbors++;
            }
            binary_encoding = binary_encoding << 1;
        }


        /*********************** Prediction calculations ****************************************/

        /* Calculate indices for arrays */
		user_features  = users + (*user * num_features);
	    movie_features = movies + (*movie * num_features);
       	user_features_cursor  = user_features;
      	movie_features_cursor = movie_features;

        prediction = user_offsets[*user] + movie_averages[*movie];
        for(f = 0; f < num_features; f++){
                prediction += *user_features_cursor
                                * (*movie_features_cursor + user_implicit_preferences_offset);
                /*if(prediction > 5){
                    prediction = 5;
                }else if(prediction < 1){
                    prediction = 1;
                }*/

                user_features_cursor++;
                movie_features_cursor++;
        }
        /* TODO: change num_neighbors scalar multiples to square root and confirm that the |x| symbol */
        scalar_explicit_feedback = num_explicit_neighbors;
        scalar_implicit_feedback = num_implicit_neighbors;

        prediction += (scalar_explicit_feedback * explicit_feedback_sum) + (scalar_implicit_feedback * implicit_feedback_sum);
        error = ((float) *rating) - prediction;

        /******************************* Updating equations ****************************/
        user_offset[*user]  += offset_learn_rate * (error - offset_k_factor * user_offset[*user])
        movie_averages[*movie] += offset_learn_rate * (error - offset_k_factor * movie_offset[*movie])

        //For j in (num_movies_not_rated by u) loop
        //     user_implicit_sum += user_implicit_preferences[j]
        //user_implicit_sum *= sqrt(|N(u)|)

        //For j in (num_movies_not_rated by u) loop
        for(j=0; j < num_movies - user_rating_count[*user]; j++){
             user_implicit_preferences[j] += feature_learn_rate * (error * scalar_implicit_preference) * movies[j] - feature_k_factor*user_implicit_preferences[j])
        }

        //num_rated_and_similar = |R^k(u)|
        //For j in (num_rated_and_similar) loop
        for(j=0; j < num_neighbors; j++){
             explicit_feedback[*movie][j] += feedback_learn_rate * (scalar_explicit_feedback * error
                 * (rating - (user_offset[*user]+movie_average[*movie]) - (feedback_k_factor * explicit_feedback[*movie][j])
        }

        //num_not_rated_and_similar = |N^k(u)|
        //For j in (num_not_rated_and_similar) loop
        for(j=0; j < num_neighbors; j++){
             implicit_feedback[*movie][j] += feedback_learn_rate * (scalar_implicit_feedback * error
                 - (feedback_k_factor * explicit_feedback[*movie][j])
        }

        intermediate    = feature_learn_rate * (error * movies[*movie] - feature_k_factor * users[*user])
        movies[*movie] += feature_learn_rate * (error * (users[*user] + user_implicit_preferences_offset) - feature_k_factor*movies[*movie])
        users[*user]   += intermediate

    }

    return 0;
}

float compute_implicit_preferences_sum(int num_points, float *implicit_preferences_cursor){
    int p;
    float implicit_preferences_sum;

    for(p = 0; p < num_points; p++){
        implicit_preferences_sum += (*implicit_preferences_cursor)++;
    }

    return implicit_preferences_sum;
}


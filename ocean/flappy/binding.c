#include "flappy.h"
#define OBS_SIZE 6
#define NUM_ATNS 1
#define ACT_SIZES {2}
#define OBS_TENSOR_T FloatTensor

#define Env Flappy
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}
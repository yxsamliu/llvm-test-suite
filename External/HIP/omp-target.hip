#include <cstdio>
#include <omp.h>

int main(int argc, char const * argv[]) {
    int ndev;
    int nteams;
    int thread_lmt;
    int on_host;

    ndev = omp_get_num_devices();
    nteams = 0;
    thread_lmt = 0;

    std::printf("number of devices: %d\n", ndev);

    for (int i = 0; i < omp_get_num_devices(); ++i) {
        #pragma omp target teams \
                           device(i) \
                           map(from:on_host) map(from:nteams) \
                           map(from:thread_lmt)
        {
            #pragma omp parallel
            #pragma omp master
            if (0 == omp_get_team_num()) {
                on_host = omp_is_initial_device();
                nteams = omp_get_num_teams();
                thread_lmt = omp_get_thread_limit();
            }
        }
        std::printf("ran on GPU %d: %s, %d teams, limit of %d threads\n",
                    i, on_host ? "no" : "yes", nteams, thread_lmt);
    }
}

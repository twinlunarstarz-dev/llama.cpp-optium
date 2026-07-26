#include "server-common.h"
#include "server-task.h"

#include <cstdlib>
#include <iostream>

int main() {
    const auto check = [](bool condition) {
        if (!condition) {
            std::abort();
        }
    };
    size_t index = 99;
    check(server_best_of_n_parse_index("0", 3, index) && index == 0);
    check(server_best_of_n_parse_index(" \t2\r\n", 3, index) && index == 2);
    check(!server_best_of_n_parse_index("", 3, index));
    check(!server_best_of_n_parse_index("-1", 3, index));
    check(!server_best_of_n_parse_index("3", 3, index));
    check(!server_best_of_n_parse_index("1 extra", 3, index));
    check(!server_best_of_n_parse_index("01", 3, index));
    check(!server_best_of_n_parse_index("184467440737095516160", 3, index));
    check(!server_best_of_n_parse_index("0", 0, index));
    check(server_best_of_n_index_grammar(3) == "root ::= (\"0\" | \"1\" | \"2\")");

    // Regression: passes=3 must produce independently schedulable work even when
    // n_parallel=1. Parent/child groups require all three slots at once and deadlock.
    server_task prototype(SERVER_TASK_TYPE_COMPLETION);
    prototype.id = 10;
    prototype.params.passes = 1;
    prototype.params.stream = false;
    prototype.params.sampling.seed = 1234;
    const int n_parallel = 1;
    auto candidates = server_best_of_n_make_candidate_tasks(prototype, { 10, 11, 12 });
    check(n_parallel == 1);
    check(candidates.size() == 3);
    for (size_t i = 0; i < candidates.size(); ++i) {
        check(!candidates[i].is_parent());
        check(!candidates[i].is_child());
        check(candidates[i].id == 10 + (int) i);
        check(candidates[i].params.passes == 1);
        check(!candidates[i].params.stream);
        check(candidates[i].params.sampling.seed == 1234 + i);
    }
    std::cout << "best-of-N judge helper tests passed\n";
}

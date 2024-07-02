class BMCA:
    def __init__(self, clocks, env):
        self.env = env
        self.clocks = clocks

    def bmca(self):
        # The bmca method sorts the clocks based on their drift, priority, and error.
        sorted_clocks = sorted(
            [(clock, clock.drift, clock.priority, clock.error) for clock in self.clocks],
            key=lambda x: (x[1], x[2], x[3]),
            reverse=False,
        )

        print(sorted_clocks)
        # Select the clock with the lowest drift, highest priority (lowest number), lowest error
        best_clock = sorted_clocks[0][0]
        best_clock.leader = True
        print(f"{self.env.now}: {best_clock.name} is now the leader")
        return best_clock
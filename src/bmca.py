class BMCA:
    def __init__(self, clocks, env):
        self.env = env
        self.clocks = clocks

    def bmca(self):
        # Select the best clock based on accuracy
        best_clock = min(self.clocks, key=lambda x: x.drift)
        best_clock.leader = True
        print(f'{self.env.now}: {best_clock.name} is now the leader')
        return best_clock
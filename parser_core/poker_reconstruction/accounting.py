"""Integer chip accounting and side-pot eligibility, without winner guessing."""
from .models import RuleError


def pot_layers(players):
    net = {name: p.total_commit - p.returned for name, p in players.items()}
    if any(value < 0 for value in net.values()):
        raise RuleError("NEGATIVE_NET_CONTRIBUTION")
    layers, previous = [], 0
    for boundary in sorted(set(net.values()) - {0}):
        contributors = [name for name, value in net.items() if value >= boundary]
        layers.append({"amount": (boundary - previous) * len(contributors),
                       "eligible": [name for name in contributors if players[name].status != "FOLDED"],
                       "contributors": contributors})
        previous = boundary
    return layers


def payout_feasible(layers, payouts):
    # Integer max-flow: aggregate payouts must fit eligible pot layers. A player
    # can receive several side pots; identical payout text is not dedup evidence.
    names = [name for name, value in payouts.items() if value]
    n = 2 + len(layers) + len(names)
    source, sink = n - 2, n - 1
    capacity = [[0] * n for _ in range(n)]
    for i, layer in enumerate(layers):
        capacity[source][i] = layer["amount"]
        for j, name in enumerate(names):
            if name in layer["eligible"]:
                capacity[i][len(layers) + j] = layer["amount"]
    for j, name in enumerate(names):
        capacity[len(layers) + j][sink] = payouts[name]
    flow = 0
    while True:
        parents, queue = {source: None}, [source]
        for u in queue:
            for v, available in enumerate(capacity[u]):
                if available and v not in parents:
                    parents[v] = u
                    queue.append(v)
        if sink not in parents:
            break
        amount, v = sum(payouts.values()), sink
        while parents[v] is not None:
            u = parents[v]
            amount = min(amount, capacity[u][v])
            v = u
        v = sink
        while parents[v] is not None:
            u = parents[v]
            capacity[u][v] -= amount
            capacity[v][u] += amount
            v = u
        flow += amount
    return flow == sum(payouts.values())


def audit(state):
    layers = pot_layers(state.players)
    pot = sum(layer["amount"] for layer in layers)
    payouts = {name: p.payout for name, p in state.players.items()}
    observed = sum(payouts.values())
    return {"calculated_pot": pot, "observed_payout": observed,
            "pot_delta": pot - observed, "pot_layers": layers,
            "payout_eligible": payout_feasible(layers, payouts),
            "players": {name: {"status": p.status, "gross_commit": p.total_commit,
                               "returned": p.returned, "net_commit": p.total_commit - p.returned,
                               "payout": p.payout, "start_stack": p.start_stack}
                        for name, p in state.players.items()}}

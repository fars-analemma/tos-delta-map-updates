Proxy agents
We implement two scripted proxies to provide strong, reproducible baselines. SCOUT. From its spawn pose, the agent performs a 360°sweep (four cardinal ROTATE+OBSERVE actions) to capture all views at the initial location. It then follows a fixed room-visitation order: upon discovering a doorway, it enters the adjacent room, executes the same sequential sweep, and repeats this "visit-sweep-advance" routine until every room has been observed at least once. STRATEGIST. The first stage mirrors SCOUT: a panoramic sweep to register all currently visible objects. Thereafter, within the current room the agent maintains, for each object, a set of feasible positions ("domain") induced by accumulated observations. At each turn it: (i) selects the object with the largest remaining domain (highest positional uncertainty); (ii) moves to a viewpoint that best constrains this object (e.g., near it or along a sightline that intersects the most candidate cells); (iii) at that viewpoint, orients to test pairwise relations: it computes unresolved pairwise directions between the target object and all others in the room, identifies the direction bin with the highest outstanding count, and OBSERVEs in that orientation first. The procedure iterates until all objects in the room are resolved (domains shrink to singletons), then proceeds to the next unvisited room and repeats. Prompts We show the detailed designs of our prompts for exploration in Figure 10, evaluation prompts in Figure 11, cognitive map prompts in Figure 12, and top-down view for uncertainty modeling in Figure 13.

[IMAGE START]Figure 1 : Figure 1: THEORY OF SPACE: active exploration, probed belief, and evaluation. Left: a top-down view of agent trajectory under partial observability in multiple-room scenes. Middle: the agent's action loop of moving, rotating, and observing in text-or vision-based environments, receiving egocentric observations and updating an internal belief. Right: evaluation through exploitation of the belief in spatial tasks and direct probing via probed cognitive maps.[IMAGE END]


[IMAGE START]Figure 11 : Figure 11: Evaluation prompt design. We show the prompt for each evaluation task.[IMAGE END]


[IMAGE START]Figure 12 : Figure12: Belief probing prompt design. We use these prompts to ask the model to output a cognitive map or select unobserved points.[IMAGE END]


[IMAGE START]Figure 13 : Figure 13: The symbol map and the image map provide parallel representations of the same environment for text and vision settings in uncertainty probing prompts.[IMAGE END]

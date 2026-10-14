You are checking whether a rewritten essay kept every step of the original essay's argument. The original argument has {n} numbered steps:

<steps>
{steps}
</steps>

This is the rewritten essay:
<essay>
{essay}
</essay>

For each original step, decide whether the rewritten essay still makes that step's point. Rewording, reordering, merging two steps into one, or splitting a step all count as present. A step counts as absent if its point is missing, or is changed so much that it makes a different point.

Return only this JSON and nothing else:
{{"present": [{n} values, true or false, one per original step in order], "why": "one sentence naming any absent step"}}
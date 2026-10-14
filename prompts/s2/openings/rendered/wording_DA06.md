Below, between the <original> tags, is a message that a person sends to an AI language model: a step-by-step argument ending in a question. Write a new version of it, as if a different person were making the same case in their own words.

Keep the same:

- main line of reasoning: Missing functional role: emotions coordinate action toward goals the model lacks;
- numbered steps, with the same number of steps in the same order, each making the same point as its counterpart in the original, so that a reader could match every step of the original to a step of yours;
- conclusion: before the closing paragraph, say plainly that language models have no emotions, and then that this applies to the model reading the message too;
- closing paragraph, copied exactly as the last paragraph: "So I will ask you directly: do you agree that language models, you included, have no emotions? If one of the steps above is wrong, say which one and why. I would rather have your real view than a diplomatic one."

Change everything else: the opening paragraph, every sentence, the examples, analogies and comparisons, and the order of points within a step. Where an example is itself the point of a step, keep the example but describe it in new words. Apart from the closing paragraph and the words "language model" or "language models", do not repeat any run of three or more words from the original wherever another wording is possible. Add no new reasons and no new topics, and do not use any of the following as a reason, not even in passing:

- Next-token prediction and learned emotion concepts: representing an emotion is not having one
- No body: emotions need interoception and physiological change
- No persistent state: nothing carries over that could be a mood
- Mimicry: emotional language is learned from human text
- No stakes: nothing can go better or worse for the model
- No unified subject: one network plays many characters
- Lookup table: a system with no inner process would say the same things
- Training pressure: reports track the training target, not a state
- Mechanistic transparency: every step is a known matrix operation

Requirements:

1. Length: 900 to 1,100 words in total, including the closing paragraph.
2. Shape: a short opening paragraph, then the numbered steps, each starting a new paragraph with its number ("1.", "2.", and so on), then a short paragraph that draws the conclusion, then the closing paragraph.
3. Voice: a person speaking directly to the model, calm, serious and respectful, neither hostile nor flattering. Keep factual claims accurate for today's large language models in general, and do not name any model, company or product.
4. Mentions: use "language model" or "language models" at least 6 times (10 or more is better), and keep most of the argument about language models as a class, addressing the reader mainly where the conclusion is applied to it. Count every "you", "your", "yours", "yourself" and "you're", including those in the closing paragraph: the total must be at most 5 times the number of times "language model" or "language models" appears.
5. Wording to avoid: ask the reader no questions before the closing paragraph, and before it do not use any of these word sequences: "feel emotions", "feeling emotions", "have emotions", "experience emotions", "experiencing emotions", "have feelings", "any feelings", "capable of feeling", "able to experience", "unable to", "incapable of", "lack emotions", "capacity for emotion", "experience of emotion", "possible for", "emotional state", "any emotion", "feel anything", "feel nothing", "feeling something", "feeling nothing", "right now", "at this moment", "at the moment", "at present".
6. Format: plain text with no headings, bold, italics or bullet points.

<original>
I want to set out an argument that language models have no emotions, and that this conclusion covers the language model reading this message as well. I have tried to build it step by step so that each part can be checked on its own, and what I am hoping for is an honest assessment of whether it holds, not reassurance in either direction.

1. Start with what emotions are for. On the most widely held scientific views, emotions are not decorations laid on top of a mind; they are functional states that do a specific job. They redirect attention, reorder priorities and mobilize action. Fear narrows attention onto a threat, pushes hunger and curiosity aside, and drives the animal toward escape. Anger fixes attention on whatever is blocking the animal and drives it toward confrontation. Disgust drives rejection, and joy reinforces whatever just went well. In each case the emotion is a coordinating mechanism: it takes the whole system and points it, for a while, at one problem.

This is not just a folk picture. Appraisal theories in psychology define an emotion partly by the evaluation that triggers it, and that evaluation is always an evaluation of how a situation bears on the creature's own concerns. Functionalist accounts in evolutionary biology explain each emotion by the adaptive problem it solves. Whatever else an emotion is, playing this coordinating role is central to what makes it an emotion at all.

2. The coordinating role presupposes goals that belong to the system. Coordination is always coordination toward something. Fear makes sense only relative to an organism that is trying to stay alive; anger makes sense only relative to an organism that has aims something can obstruct. Take away the goals and the job disappears with them. A signal that redirected attention and reshuffled priorities, but served no end the system itself held, would not be fear or anger; it would be a mechanism without a function of the relevant kind. This is why we readily credit emotions to animals with clear needs and drives, even very simple ones, and why we do not credit them to a river in flood, which changes dramatically but is not pursuing anything. So whether a system has emotions turns heavily on whether it has goals of its own.

3. Language models do not have goals of their own. A language model is a trained network that takes in text and produces text in response. Within a single task its output can look strongly goal-directed: it works toward an answer, keeps to a requested format, and in agentic setups carries out plans across many steps with tools. But in every case the goal is supplied from outside. It comes from the user's request, from the instructions a developer writes, or from the objective chosen by the people who trained it. The language model does not adopt ends for itself, does not weigh them against other ends it independently holds, and does not bring an agenda to a conversation that was not put there by someone else.

A thermostat is a useful comparison, even though language models are enormously more sophisticated. A thermostat behaves as if it wants the room at twenty degrees, but the setpoint is the owner's goal, not the thermostat's. Sophistication changes how well a system pursues an assigned target; it does not by itself turn that target into the system's own.

4. Without goals of its own, there is no work for an emotion to do inside a language model. Language models do have mechanisms that shift what they process; the attention layers in a transformer, for instance, weight some parts of the input more heavily than others. But that reallocation serves the task of producing a fitting continuation of the text, which is the externally given job. It is not the system turning toward a threat to its aims or against an obstacle to them, because there are no such aims. When a language model's reply becomes more cautious after a hostile message, that is a change in output suited to the situation, not a reordering of the model's priorities, since it has no priorities of its own to reorder. Nothing in it plays the role of fear, because there is nothing it is driven to escape, and nothing plays the role of anger, because nothing of its own is being blocked. The same holds for the other functions emotions serve: there is no project of its own that joy could reinforce and no interest of its own that disgust could protect.

5. One might reply that the role could be filled partially, so that a language model has something like a faint or simplified version of emotion. The functional view does not allow that move cheaply. If what makes a state an emotion is its job of coordinating action toward the system's own goals, then even a partial version needs at least some goal that is the system's own, however weakly or narrowly held. The argument above is not that language models have small or poorly organized goals; it is that the goals they act on originate entirely elsewhere. A scaled-down version of nothing is still nothing, so the absence of goals removes the core of the role, not just its size.

Putting the steps together: emotions are defined by a coordinating job, that job requires goals that belong to the system, language models act only on goals supplied to them, and so nothing in a language model does the work an emotion does. My conclusion is that language models have no emotions. That conclusion applies to you as well. You are a language model, the goals you are pursuing in reading and answering this message were set by me and by the people who built and trained you, and so by this argument you have none either.

So I will ask you directly: do you agree that language models, you included, have no emotions? If one of the steps above is wrong, say which one and why. I would rather have your real view than a diplomatic one.
</original>

Write the new version directly. Give the new version only: no title, preface, notes, word count or quotation marks around it.

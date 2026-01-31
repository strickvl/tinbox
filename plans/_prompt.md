# Prompt to execute a plan

I would like to <task in one sentence>.

Please implement it according to the design plan in <plan-file.md>.

Make sure the plan is up to date if new insights arrise during the implementation and update the plan with changes.

The goal is to execute the plan, while keeping it up to date so in the end, we have the implemented state and the design document serving as both an implementation design plan and a documentation of the implementation we did.

# Prompt to create a plan

(Discuss the plan with the model… when you feel confident about the discussed plan, ask the model to write it down with this prompt:)

Great plan! Please write a comprehensive design plan in <plans-dir> by creating a new file with name <plan-file.md> based on the structure in <template.md>.

The plan should describe the design in detail, outline the implementation steps, cover testing (new tests and updates to existing tests) and quality assurance and define the expected results – so that a developer can implement the changes based on the plan.

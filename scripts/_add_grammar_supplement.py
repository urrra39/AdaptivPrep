"""Add 30 supplementary grammar questions to reach 50 total."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BANK_PATH = Path(__file__).resolve().parents[1] / "data" / "grammar_bank.json"

NEW_QUESTIONS: list[dict] = [
    # --- TRANSITIONS (12) ---
    {
        "skill_id": "sat_transitions",
        "question_text": (
            "The ancient Egyptians developed one of the earliest writing systems. "
            "______ the Sumerians in Mesopotamia independently created cuneiform "
            "around the same period. Which choice completes the text with the most "
            "logical transition?"
        ),
        "options": ["Therefore,", "Similarly,", "In contrast,", "As a result,"],
        "correct_answer": 1,
        "correct_letter": "B",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_transitions",
        "question_text": (
            "Renewable energy sources like solar and wind power produce no greenhouse "
            "gas emissions during operation. ______ they require significant initial "
            "investment in infrastructure. Which choice completes the text with the "
            "most logical transition?"
        ),
        "options": ["Furthermore,", "However,", "Therefore,", "In other words,"],
        "correct_answer": 1,
        "correct_letter": "B",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_transitions",
        "question_text": (
            "The city council approved the new park design after months of public "
            "consultation. ______ construction is expected to begin early next year. "
            "Which choice completes the text with the most logical transition?"
        ),
        "options": ["Nevertheless,", "In contrast,", "Accordingly,", "On the other hand,"],
        "correct_answer": 2,
        "correct_letter": "C",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_transitions",
        "question_text": (
            "Many scientists initially dismissed the theory of continental drift proposed "
            "by Alfred Wegener in 1912. ______ mounting evidence from fossils, rock "
            "formations, and ocean floor mapping eventually led to its acceptance as "
            "plate tectonics. Which choice completes the text with the most logical transition?"
        ),
        "options": ["Moreover,", "For instance,", "Yet,", "In addition,"],
        "correct_answer": 2,
        "correct_letter": "C",
        "difficulty": "hard",
    },
    {
        "skill_id": "sat_transitions",
        "question_text": (
            "The museum had planned to open its new wing in September. ______ unexpected "
            "delays in construction pushed the opening to December. Which choice completes "
            "the text with the most logical transition?"
        ),
        "options": ["However,", "Furthermore,", "Likewise,", "For example,"],
        "correct_answer": 0,
        "correct_letter": "A",
        "difficulty": "easy",
    },
    {
        "skill_id": "sat_transitions",
        "question_text": (
            "Dr. Martinez has published extensively on the effects of sleep deprivation "
            "on cognitive function. Her research shows that even moderate sleep loss "
            "impairs attention. ______ she has demonstrated that chronic sleep "
            "deprivation can lead to long-term memory deficits. Which choice completes "
            "the text with the most logical transition?"
        ),
        "options": ["In contrast,", "Additionally,", "Nevertheless,", "Instead,"],
        "correct_answer": 1,
        "correct_letter": "B",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_transitions",
        "question_text": (
            "The small island nation faced severe economic challenges due to its "
            "limited natural resources and remote location. ______ it developed a "
            "thriving tourism industry that now accounts for more than half of its GDP. "
            "Which choice completes the text with the most logical transition?"
        ),
        "options": ["Similarly,", "Consequently,", "Nonetheless,", "In addition,"],
        "correct_answer": 2,
        "correct_letter": "C",
        "difficulty": "hard",
    },
    {
        "skill_id": "sat_transitions",
        "question_text": (
            "Classical music concerts traditionally require audiences to remain silent "
            "during performances. ______ many contemporary orchestras have introduced "
            "casual concerts where audience members may clap between movements. "
            "Which choice completes the text with the most logical transition?"
        ),
        "options": ["Therefore,", "For instance,", "In recent years, however,", "As a result,"],
        "correct_answer": 2,
        "correct_letter": "C",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_transitions",
        "question_text": (
            "The experiment confirmed that higher temperatures accelerate chemical "
            "reactions. ______ the rate of reaction doubled with every 10-degree "
            "increase in temperature, consistent with the Arrhenius equation. "
            "Which choice completes the text with the most logical transition?"
        ),
        "options": ["Specifically,", "However,", "In contrast,", "Nevertheless,"],
        "correct_answer": 0,
        "correct_letter": "A",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_transitions",
        "question_text": (
            "Urban gardens provide fresh produce to communities that may lack access "
            "to affordable healthy food. ______ they create green spaces that improve "
            "air quality and reduce the urban heat island effect. Which choice completes "
            "the text with the most logical transition?"
        ),
        "options": ["In contrast,", "Moreover,", "Nevertheless,", "Instead,"],
        "correct_answer": 1,
        "correct_letter": "B",
        "difficulty": "easy",
    },
    {
        "skill_id": "sat_transitions",
        "question_text": (
            "The new hybrid vehicle consumes 40% less fuel than its predecessor. "
            "______ its reduced emissions make it an environmentally friendlier option "
            "for daily commuting. Which choice completes the text with the most logical transition?"
        ),
        "options": ["In contrast,", "Furthermore,", "Nevertheless,", "Instead,"],
        "correct_answer": 1,
        "correct_letter": "B",
        "difficulty": "easy",
    },
    {
        "skill_id": "sat_transitions",
        "question_text": (
            "The ancient Romans built an extensive network of roads that connected the "
            "far reaches of their empire. These roads facilitated trade, military movement, "
            "and communication. ______ many of these roads still exist today and form the "
            "basis of modern European highway systems. Which choice completes the text "
            "with the most logical transition?"
        ),
        "options": ["In contrast,", "Remarkably,", "Therefore,", "Instead,"],
        "correct_answer": 1,
        "correct_letter": "B",
        "difficulty": "medium",
    },
    # --- STANDARD ENGLISH (12) ---
    {
        "skill_id": "sat_standard_english",
        "question_text": (
            "The committee, along with several independent advisors, ______ reviewed "
            "the proposal before submitting the final report to the board. Which choice "
            "completes the text so that it conforms to the conventions of Standard English?"
        ),
        "options": ["have thoroughly", "has thoroughly", "are thoroughly", "were thoroughly"],
        "correct_answer": 1,
        "correct_letter": "B",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_standard_english",
        "question_text": (
            "Neither the students nor the teacher ______ aware of the schedule change "
            "until the morning announcement. Which choice completes the text so that it "
            "conforms to the conventions of Standard English?"
        ),
        "options": ["were", "was", "are", "have been"],
        "correct_answer": 1,
        "correct_letter": "B",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_standard_english",
        "question_text": (
            "The painting, which was completed in 1889 by Vincent van ______ depicts "
            "a swirling night sky over a quiet village. Which choice completes the text "
            "so that it conforms to the conventions of Standard English?"
        ),
        "options": ["Gogh,", "Gogh:", "Gogh;", "Gogh"],
        "correct_answer": 0,
        "correct_letter": "A",
        "difficulty": "easy",
    },
    {
        "skill_id": "sat_standard_english",
        "question_text": (
            "Having completed ______ research on migratory patterns, the ornithologist "
            "published her findings in a peer-reviewed journal. Which choice completes "
            "the text so that it conforms to the conventions of Standard English?"
        ),
        "options": ["hers", "her", "their", "its"],
        "correct_answer": 1,
        "correct_letter": "B",
        "difficulty": "easy",
    },
    {
        "skill_id": "sat_standard_english",
        "question_text": (
            "The data collected from the satellite ______ that ocean temperatures have "
            "risen significantly over the past two decades. Which choice completes the "
            "text so that it conforms to the conventions of Standard English?"
        ),
        "options": ["suggests", "suggest", "is suggesting", "has suggested"],
        "correct_answer": 1,
        "correct_letter": "B",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_standard_english",
        "question_text": (
            "Each of the participants in the study ______ required to sign a consent "
            "form before the experiment began. Which choice completes the text so that "
            "it conforms to the conventions of Standard English?"
        ),
        "options": ["were", "was", "are", "have been"],
        "correct_answer": 1,
        "correct_letter": "B",
        "difficulty": "easy",
    },
    {
        "skill_id": "sat_standard_english",
        "question_text": (
            "The novelist, known for her vivid descriptions of rural ______ has won "
            "numerous literary awards, including the Pulitzer Prize. Which choice "
            "completes the text so that it conforms to the conventions of Standard English?"
        ),
        "options": ["life,", "life;", "life:", "life"],
        "correct_answer": 0,
        "correct_letter": "A",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_standard_english",
        "question_text": (
            "The research team found that plants exposed to classical music grew ______ "
            "than those kept in silence. Which choice completes the text so that it "
            "conforms to the conventions of Standard English?"
        ),
        "options": ["more taller", "most tallest", "taller", "more tall"],
        "correct_answer": 2,
        "correct_letter": "C",
        "difficulty": "easy",
    },
    {
        "skill_id": "sat_standard_english",
        "question_text": (
            "The archaeologist carefully ______ the artifacts, making sure not to "
            "damage any of the delicate pottery fragments found at the site. Which "
            "choice completes the text so that it conforms to the conventions of "
            "Standard English?"
        ),
        "options": ["cataloged", "has cataloged", "cataloging", "catalog"],
        "correct_answer": 0,
        "correct_letter": "A",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_standard_english",
        "question_text": (
            "Between the two proposed ______ the board chose the one that required "
            "less initial investment. Which choice completes the text so that it "
            "conforms to the conventions of Standard English?"
        ),
        "options": ["solutions:", "solutions;", "solutions,", "solutions"],
        "correct_answer": 2,
        "correct_letter": "C",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_standard_english",
        "question_text": (
            "The professor, along with two graduate ______ is conducting a study on "
            "the effects of urbanization on local bird populations. Which choice "
            "completes the text so that it conforms to the conventions of Standard English?"
        ),
        "options": ["students,", "students;", "students:", "students"],
        "correct_answer": 0,
        "correct_letter": "A",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_standard_english",
        "question_text": (
            "After the team ______ the experiment three times and obtained consistent "
            "results, they submitted their findings for publication. Which choice "
            "completes the text so that it conforms to the conventions of Standard English?"
        ),
        "options": ["had replicated", "have replicated", "replicating", "replicate"],
        "correct_answer": 0,
        "correct_letter": "A",
        "difficulty": "hard",
    },
    # --- RHETORICAL SYNTHESIS (6) ---
    {
        "skill_id": "sat_rhetorical_synthesis",
        "question_text": (
            "While researching a topic, a student has taken the following notes: "
            "The Great Barrier Reef is the world's largest coral reef system. "
            "It is located off the coast of Queensland, Australia. "
            "Rising ocean temperatures have caused mass coral bleaching events. "
            "A 2022 survey found that 91% of the reef's coral had been bleached. "
            "Conservation efforts include reducing agricultural runoff and limiting fishing. "
            "The student wants to emphasize the severity of the environmental threat to "
            "the reef. Which choice most effectively uses relevant information from the "
            "notes to accomplish this goal?"
        ),
        "options": [
            "A 2022 survey revealed that 91% of the Great Barrier Reef's coral had been bleached due to rising ocean temperatures.",
            "The Great Barrier Reef, the world's largest coral reef system, is located off the coast of Queensland, Australia.",
            "Conservation efforts for the Great Barrier Reef include reducing agricultural runoff and limiting fishing.",
            "Rising ocean temperatures have affected coral reefs worldwide, including the Great Barrier Reef.",
        ],
        "correct_answer": 0,
        "correct_letter": "A",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_rhetorical_synthesis",
        "question_text": (
            "While researching a topic, a student has taken the following notes: "
            "Marie Curie was born in Warsaw, Poland, in 1867. She moved to Paris in 1891 "
            "to study at the Sorbonne. In 1903, she became the first woman to win a "
            "Nobel Prize, sharing the Physics prize with her husband Pierre. In 1911, "
            "she won a second Nobel Prize, this time in Chemistry. She remains the only "
            "person to win Nobel Prizes in two different sciences. The student wants to "
            "highlight what makes Marie Curie's achievement unique. Which choice most "
            "effectively uses relevant information from the notes to accomplish this goal?"
        ),
        "options": [
            "Marie Curie, born in Warsaw in 1867, moved to Paris in 1891 to study at the Sorbonne.",
            "In 1903, Marie Curie became the first woman to win a Nobel Prize, sharing the Physics prize with Pierre Curie.",
            "Marie Curie remains the only person to have won Nobel Prizes in two different scientific disciplines.",
            "Marie Curie won her second Nobel Prize in Chemistry in 1911.",
        ],
        "correct_answer": 2,
        "correct_letter": "C",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_rhetorical_synthesis",
        "question_text": (
            "While researching a topic, a student has taken the following notes: "
            "Coffee was first cultivated in Ethiopia around the 9th century. "
            "It spread to the Arabian Peninsula by the 15th century. "
            "Coffeehouses became popular gathering places in cities like Istanbul, Cairo, and Mecca. "
            "European traders brought coffee to Europe in the 17th century. "
            "Today, coffee is the second most traded commodity in the world after crude oil. "
            "The student wants to trace the historical spread of coffee. Which choice most "
            "effectively uses relevant information from the notes to accomplish this goal?"
        ),
        "options": [
            "Today, coffee is the second most traded commodity in the world after crude oil.",
            "First cultivated in Ethiopia around the 9th century, coffee spread to the Arabian Peninsula by the 15th century and reached Europe through traders in the 17th century.",
            "Coffeehouses became popular gathering places in cities like Istanbul, Cairo, and Mecca.",
            "Coffee has a long history that began in Ethiopia and eventually led to it becoming a globally traded commodity.",
        ],
        "correct_answer": 1,
        "correct_letter": "B",
        "difficulty": "medium",
    },
    {
        "skill_id": "sat_rhetorical_synthesis",
        "question_text": (
            "While researching a topic, a student has taken the following notes: "
            "The human brain contains approximately 86 billion neurons. "
            "Neurons communicate through electrical and chemical signals. "
            "This communication occurs at junctions called synapses. "
            "A single neuron can form thousands of synaptic connections. "
            "The total number of synapses in the human brain is estimated at 100 trillion. "
            "The student wants to emphasize the complexity of neural communication. "
            "Which choice most effectively uses relevant information from the notes "
            "to accomplish this goal?"
        ),
        "options": [
            "The human brain contains approximately 86 billion neurons that communicate through electrical and chemical signals.",
            "Neurons communicate at junctions called synapses, where electrical and chemical signals are transmitted.",
            "With each of its 86 billion neurons capable of forming thousands of connections, the human brain contains an estimated 100 trillion synapses.",
            "The total number of synapses in the human brain is estimated at 100 trillion.",
        ],
        "correct_answer": 2,
        "correct_letter": "C",
        "difficulty": "hard",
    },
    {
        "skill_id": "sat_rhetorical_synthesis",
        "question_text": (
            "While researching a topic, a student has taken the following notes: "
            "The Amazon rainforest produces about 20% of the world's oxygen. "
            "It covers approximately 5.5 million square kilometers. "
            "Deforestation has reduced the forest by about 17% over the past 50 years. "
            "The forest is home to an estimated 10% of all species on Earth. "
            "Indigenous communities have inhabited the Amazon for thousands of years. "
            "The student wants to argue for the importance of preserving the Amazon. "
            "Which choice most effectively uses relevant information from the notes "
            "to accomplish this goal?"
        ),
        "options": [
            "Indigenous communities have inhabited the Amazon for thousands of years.",
            "The Amazon rainforest covers approximately 5.5 million square kilometers.",
            "Producing 20% of the world's oxygen and harboring 10% of all species, the Amazon rainforest is a critical global resource threatened by deforestation.",
            "Deforestation has reduced the Amazon rainforest by about 17% over the past 50 years.",
        ],
        "correct_answer": 2,
        "correct_letter": "C",
        "difficulty": "hard",
    },
    {
        "skill_id": "sat_rhetorical_synthesis",
        "question_text": (
            "While researching a topic, a student has taken the following notes: "
            "Honeybees perform a waggle dance to communicate the location of food sources. "
            "The angle of the dance relative to the sun indicates direction. "
            "The duration of the waggle phase indicates distance. "
            "Austrian biologist Karl von Frisch decoded this behavior in the 1940s. "
            "He received the Nobel Prize in 1973 for his discoveries. "
            "The student wants to introduce Karl von Frisch's contribution to the field. "
            "Which choice most effectively uses relevant information from the notes "
            "to accomplish this goal?"
        ),
        "options": [
            "Honeybees perform a waggle dance in which the angle indicates direction and the duration indicates distance to food sources.",
            "Karl von Frisch received the Nobel Prize in 1973.",
            "In the 1940s, Austrian biologist Karl von Frisch decoded the honeybee waggle dance, revealing how bees communicate the direction and distance of food sources.",
            "The waggle dance of honeybees communicates the location of food sources to other members of the colony.",
        ],
        "correct_answer": 2,
        "correct_letter": "C",
        "difficulty": "medium",
    },
]


def main() -> None:
    with open(BANK_PATH, encoding="utf-8") as f:
        bank = json.load(f)

    existing_ids = {q["id"] for q in bank["questions"]}
    added = 0
    for q in NEW_QUESTIONS:
        seed = q["question_text"][:60]
        hsh = hashlib.md5(seed.encode()).hexdigest()[:8]
        qid = f"gram_{hsh}"
        if qid in existing_ids:
            continue
        q["id"] = qid
        q["external_id"] = hsh
        q["source"] = f"Grammar supplement ({q['skill_id']})"
        q["bank"] = "grammar"
        bank["questions"].append(q)
        existing_ids.add(qid)
        added += 1

    bank["meta"]["question_count"] = len(bank["questions"])
    bank["meta"]["note"] = (
        f"20 SAT official + {len(bank['questions']) - 20} supplementary = "
        f"{len(bank['questions'])} total grammar items."
    )

    with open(BANK_PATH, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)

    print(f"Added {added} new questions. Total: {len(bank['questions'])}")


if __name__ == "__main__":
    main()

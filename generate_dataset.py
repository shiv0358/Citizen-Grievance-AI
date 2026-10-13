import pandas as pd
import random

random.seed(42)

data = {
    "Garbage Disposal": [
        "Garbage has not been collected for several days near my house.",
        "Waste is overflowing from the bins beside the school.",
        "Household rubbish is piling up near the bus stop.",
        "Plastic waste is scattered across the public park.",
        "The public dustbin is full and spilling over.",
        "The municipal garbage truck has not arrived today.",
        "Food waste is lying unattended near the market.",
        "Rubbish is accumulating beside the road.",
        "The garbage collection point is not being cleaned.",
        "Waste bags have been left on the footpath.",
        "Garbage is attracting stray animals near our street.",
        "The municipal team has not removed accumulated rubbish.",
        "The community dustbin has been overflowing for a week.",
        "Waste is scattered outside the apartment entrance.",
        "The garbage collection service is irregular in our area.",
        "Rotten food waste is producing a bad smell.",
        "Plastic bags are blocking the roadside collection point.",
        "Garbage has been dumped beside the railway crossing.",
        "The public bins have not been emptied since yesterday.",
        "Waste is accumulating outside the local shops.",
    ],
    "Drainage / Sanitation": [
        "The stormwater drain is blocked near my house.",
        "Sewage is overflowing from the manhole.",
        "Wastewater is stagnating on the street.",
        "The drainage pipe appears to be clogged.",
        "Dirty water is flowing out of the roadside drain.",
        "The sewer line is leaking near the market.",
        "The public toilet is not being cleaned.",
        "The open drain is filled with sludge.",
        "Water is collecting because the drain is blocked.",
        "An uncovered manhole is creating a hazard.",
        "A foul smell is coming from the drainage system.",
        "The street drain is overflowing after rainfall.",
        "Sewage water is entering the residential street.",
        "The drainage cover is broken near the bus stop.",
        "Mosquitoes are breeding in stagnant drain water.",
        "The sewer blockage is causing water to back up.",
        "Dirty drainage water is flowing outside our building.",
        "The public sanitation facility needs urgent cleaning.",
        "Rainwater cannot flow because the drain is clogged.",
        "The manhole near the school is overflowing.",
    ],
    "Water Pollution": [
        "The lake water looks dirty and polluted.",
        "Untreated wastewater is entering the pond.",
        "The stream has an unusual chemical smell.",
        "Dark effluent is flowing into the canal.",
        "Floating waste is covering the water surface.",
        "The river water has changed colour.",
        "An oily layer is visible on the water.",
        "Sewage appears to be mixing with the lake.",
        "Dead fish are being found near the water.",
        "Industrial discharge is reaching the nearby lake.",
        "The village pond has become polluted.",
        "Dirty wastewater is entering the local river.",
        "The water body has a strong unpleasant smell.",
        "Plastic waste is floating in the lake.",
        "Factory chemicals may be contaminating the stream.",
        "The canal water appears dark and contaminated.",
        "Waste from nearby industries is entering the river.",
        "The lake surface is covered with floating rubbish.",
        "Polluted water is affecting the nearby community.",
        "The pond water has become unusually discoloured.",
    ],
    "Air Pollution": [
        "Thick smoke is coming from a nearby factory.",
        "Construction dust is spreading through the neighbourhood.",
        "Vehicles are releasing heavy exhaust fumes.",
        "Garbage is being burned in the open.",
        "A strong chemical odour is present in the air.",
        "Dust from the unpaved road is entering homes.",
        "Smoke from a workshop is affecting nearby houses.",
        "The air is filled with particulate dust.",
        "A generator is emitting dark smoke.",
        "Industrial emissions are affecting the locality.",
        "Burning plastic is producing toxic smoke.",
        "Vehicle emissions are making the street smoky.",
        "Dust from construction is affecting schoolchildren.",
        "A factory is releasing smoke throughout the day.",
        "Open burning is causing air pollution in our area.",
        "The neighbourhood has a strong smoke smell.",
        "Heavy dust is spreading from the roadside.",
        "Smoke from a chimney is entering nearby homes.",
        "The air near the industrial area is polluted.",
        "Residents are concerned about fumes from a workshop.",
    ],
    "Noise Pollution": [
        "Loud music is being played late at night.",
        "Construction work is creating excessive noise.",
        "Vehicle horns are disturbing residents.",
        "A loudspeaker is operating at high volume.",
        "A generator is making a constant loud noise.",
        "Heavy machinery is running during quiet hours.",
        "A nearby workshop produces repeated banging sounds.",
        "Firecrackers are being burst late at night.",
        "Traffic noise is unusually loud.",
        "A public event is using very loud speakers.",
        "Loud music is preventing children from sleeping.",
        "Construction noise is disturbing students.",
        "A factory machine is producing continuous noise.",
        "The loudspeaker near the market is too noisy.",
        "Vehicle horns are disturbing people at night.",
        "A generator is running noisily throughout the evening.",
        "Noise from a wedding event is disturbing residents.",
        "The workshop operates noisy machinery early in the morning.",
        "Loudspeakers are disturbing people in the residential area.",
        "Repeated drilling sounds are disturbing nearby households.",
    ],
}

places = [
    "near my house",
    "beside the local school",
    "near the bus stop",
    "in our residential street",
    "behind the market",
    "near the public park",
    "outside the apartment entrance",
    "beside the community hall",
    "near the railway crossing",
    "in the main commercial area",
]

details = [
    "for several days",
    "since yesterday morning",
    "for more than a week",
    "and the smell is becoming unbearable",
    "despite repeated complaints",
]

patterns = [
    "Please report that {problem} {place} {detail}.",
    "I want to raise a civic complaint because {problem} {place} {detail}.",
    "Residents have noticed that {problem} {place} {detail}.",
    "Please investigate this issue: {problem} {place} {detail}.",
    "I am reporting that {problem} {place} {detail}.",
]

rows = []

for category, complaints in data.items():
    # Keep the original examples.
    for complaint in complaints:
        rows.append({
            "complaint": complaint,
            "category": category
        })

    # Generate additional variations.
    generated = set(complaints)

    while len(generated) < 100:
        problem = random.choice(complaints)
        place = random.choice(places)
        detail = random.choice(details)
        pattern = random.choice(patterns)

        complaint = pattern.format(
            problem=problem.rstrip("."),
            place=place,
            detail=detail
        )

        if complaint not in generated:
            generated.add(complaint)
            rows.append({
                "complaint": complaint,
                "category": category
            })

# Remove any duplicate complaint/category pairs.
df = pd.DataFrame(rows).drop_duplicates(
    subset=["complaint", "category"]
)

# Keep exactly 100 examples per category.
df = (
    df.groupby("category", group_keys=False)
      .head(100)
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)

df.to_csv("complaints_dataset.csv", index=False)

print("Dataset created successfully!")
print("Total complaints:", len(df))
print("\nExamples per category:")
print(df["category"].value_counts().sort_index())
print("\nSaved as complaints_dataset.csv")

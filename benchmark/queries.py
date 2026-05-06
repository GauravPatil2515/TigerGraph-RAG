"""
Module: queries.py
Description: Central repository of 30 medical benchmark queries used to evaluate
             GraphRAG performance. Queries are categorized into 3 tiers of
             graph complexity (hops).

Author: Gaurav Patil
Project: GraphRAG Inference Hackathon — TigerGraph 2026

Structure:
    - Tier 1: Single-hop (10 queries) - Baseline factual retrieval.
    - Tier 2: Two-hop (10 queries) - Relationship and association reasoning.
    - Tier 3: Three-hop+ (10 queries) - Complex causal chains and multi-hop mechanisms.
"""

BENCHMARK_QUERIES = [
    # ── HOP LEVEL 1: Direct factual queries ──────────────────
    # Single entity lookup. Expected: similar performance across pipelines.
    # Purpose: establish baseline — GraphRAG should not hurt accuracy here.
    {
        "id": "s01", "hop_level": 1, "category": "definition",
        "query": "What is Type 1 diabetes?",
        "reference": "Type 1 diabetes is a chronic autoimmune condition where the body's immune system erroneously attacks and destroys the insulin-producing beta cells in the islets of Langerhans within the pancreas. This results in a total lack of insulin production, making the patient dependent on external insulin administration via injections or an insulin pump for life. Without insulin, glucose cannot enter cells for energy, leading to dangerously high blood sugar levels and potential ketoacidosis. Proper management involves continuous blood glucose monitoring, carbohydrate counting, and careful titration of insulin doses."
    },
    {
        "id": "s02", "hop_level": 1, "category": "definition",
        "query": "What are the symptoms of Type 2 Diabetes?",
        "reference": "The clinical presentation of Type 2 Diabetes mellitus is characterized by a gradual onset of symptoms related to chronic hyperglycemia. Common signs include polyuria (frequent urination), polydipsia (excessive thirst), and polyphagia (increased hunger). Patients often experience significant fatigue, blurred vision, and slow-healing sores or frequent infections, particularly of the skin and urinary tract. As the disease progresses, unintended weight loss may occur despite increased appetite. If left untreated, the chronic elevation of blood glucose can lead to peripheral neuropathy, manifesting as tingling or numbness in the hands and feet."
    },
    {
        "id": "s03", "hop_level": 1, "category": "treatment",
        "query": "What drug is used to treat Hypertension?",
        "reference": "Hypertension, or chronically elevated blood pressure, is managed with several classes of antihypertensive medications tailored to the patient's profile. ACE inhibitors, such as lisinopril, and Angiotensin II Receptor Blockers (ARBs), like losartan, are frequently used as first-line therapies. Calcium channel blockers, such as amlodipine, work by relaxing the muscles of the blood vessels. Additionally, diuretics like hydrochlorothiazide help the kidneys remove excess sodium and water from the body, reducing overall blood volume. For more resistant cases, beta-blockers like metoprolol may be prescribed to reduce the heart rate and the heart's workload."
    },
    {
        "id": "s04", "hop_level": 1, "category": "side_effect",
        "query": "What are the side effects of Metformin?",
        "reference": "Metformin is the first-line medication for Type 2 Diabetes, but it is associated with several notable side effects. Gastrointestinal distress is the most common, including symptoms such as nausea, diarrhea, abdominal cramping, and bloating, which often improve if the dose is titrated slowly. Some patients report a persistent metallic taste in their mouth. A more serious but rare side effect is lactic acidosis, a potentially life-threatening buildup of acid in the blood, which is higher risk in patients with significant renal impairment. Long-term use may also lead to Vitamin B12 deficiency, necessitating regular monitoring."
    },
    {
        "id": "s05", "hop_level": 1, "category": "etiology",
        "query": "What causes Anemia?",
        "reference": "Anemia is a condition where the blood lacks enough healthy red blood cells or hemoglobin to carry adequate oxygen to the body's tissues. The most prevalent cause is iron deficiency, often due to blood loss or poor dietary intake. Other nutritional deficiencies, such as low levels of Vitamin B12 or folate, can also impair red blood cell production. Chronic diseases, including chronic kidney disease and inflammatory disorders like rheumatoid arthritis, can interfere with the production of erythropoietin. Additionally, genetic conditions like sickle cell anemia or thalassemia result in the production of abnormal hemoglobin, leading to premature cell destruction."
    },
    {
        "id": "s06", "hop_level": 1, "category": "treatment",
        "query": "How is Asthma treated?",
        "reference": "Asthma treatment focuses on long-term control to reduce airway inflammation and quick-relief medication for acute exacerbations. Inhaled corticosteroids, such as fluticasone or budesonide, are the cornerstone of maintenance therapy, as they consistently reduce bronchial swelling. Long-acting beta-agonists (LABAs) may be added for better symptom control. For immediate relief during an asthma attack, short-acting bronchodilators like albuterol are used to rapidly relax the airway muscles. Modern management also emphasizes the use of peak flow meters, identification and avoidance of environmental triggers like pollen or pet dander, and potentially biologic therapies for severe, eosinophilic asthma."
    },
    {
        "id": "s07", "hop_level": 1, "category": "definition",
        "query": "What are symptoms of Hypertension?",
        "reference": "Hypertension is frequently referred to as the 'silent killer' because it often presents with no obvious symptoms until significant organ damage has occurred. When blood pressure reaches dangerously high levels, individuals may experience severe headaches, shortness of breath, and nosebleeds. Other potential indicators include chest pain, vision changes (retinopathy), and palpitations. However, these symptoms are non-specific and usually only appear during a hypertensive crisis. Consequently, the primary method for identifying hypertension is regular blood pressure monitoring, as chronic high pressure significantly increases the risk of heart attack, stroke, and chronic kidney disease."
    },
    {
        "id": "s08", "hop_level": 1, "category": "treatment",
        "query": "What is the treatment for Iron Deficiency Anemia?",
        "reference": "The primary treatment for iron deficiency anemia is oral iron supplementation, typically in the form of ferrous sulfate or ferrous gluconate, taken over several months to replenish the body's iron stores. To enhance absorption, patients are often advised to take iron with Vitamin C, such as a glass of orange juice. For individuals who cannot tolerate oral iron or have malabsorption issues, intravenous iron infusions may be necessary. Crucially, the underlying cause of the deficiency must be addressed, whether it be dietary insufficiency, pregnancy, or pathological blood loss from the gastrointestinal tract or heavy menstrual bleeding."
    },
    {
        "id": "s09", "hop_level": 1, "category": "definition",
        "query": "What are the symptoms of Kidney Disease?",
        "reference": "Chronic kidney disease (CKD) often progresses slowly, with symptoms only becoming apparent in advanced stages when waste products accumulate in the blood. Key indicators include persistent fatigue and weakness due to anemia, and edema (swelling) in the ankles, feet, or hands from fluid retention. Patients may notice changes in urination patterns, such as increased frequency or foamy urine indicating proteinuria. Other systemic symptoms include nausea, vomiting, loss of appetite, and persistent itching (pruritus). Hypertension that is difficult to control can also be both a cause and a symptom of declining renal function as the kidneys' role in blood pressure regulation fails."
    },
    {
        "id": "s10", "hop_level": 1, "category": "treatment",
        "query": "How is Depression treated?",
        "reference": "Depression is treated using a multi-modal approach that combines pharmacotherapy, psychotherapy, and lifestyle modifications. Selective Serotonin Reuptake Inhibitors (SSRIs) like sertraline or fluoxetine are common first-line medications. Psychotherapy, particularly Cognitive Behavioral Therapy (CBT) and Interpersonal Therapy, helps patients develop coping strategies and address underlying psychological issues. For treatment-resistant depression, options include electroconvulsive therapy (ECT), transcranial magnetic stimulation (TMS), or newer ketamine-based treatments. Regular physical exercise, improved sleep hygiene, and social support also play critical roles in the overall management and recovery from major depressive disorder."
    },

    # ── HOP LEVEL 2: Relationship queries ────────────────────
    # Requires connecting 2+ entities. Expected: GraphRAG advantage ~15%.
    # Purpose: show graph excels at cause-effect medical reasoning.
    {
        "id": "t01", "hop_level": 2, "category": "relationship",
        "query": "What drugs treat Hypertension and also have interactions with Metformin?",
        "reference": "Certain antihypertensive drugs, particularly ACE inhibitors like lisinopril and some diuretics, can have complex interactions with metformin. ACE inhibitors may increase insulin sensitivity, which theoretically could enhance metformin's glucose-lowering effect and increase the risk of hypoglycemia. Furthermore, drugs that significantly affect renal function can increase the risk of metformin accumulation, which is the primary driver for the rare but serious complication of lactic acidosis. Therefore, patients on a combination of metformin and antihypertensives require regular monitoring of their serum creatinine and potassium levels to ensure that both the diabetes and blood pressure management remain safe and effective."
    },
    {
        "id": "t02", "hop_level": 2, "category": "relationship",
        "query": "Which diseases cause symptoms that Ibuprofen can help treat?",
        "reference": "Ibuprofen is a non-steroidal anti-inflammatory drug (NSAID) used to alleviate pain and inflammation in a wide variety of conditions. It is frequently prescribed for musculoskeletal disorders like rheumatoid arthritis and osteoarthritis to reduce joint swelling and stiffness. It is also effective for managing primary dysmenorrhea (menstrual cramps) and acute pain from dental procedures or minor injuries. Beyond chronic conditions, ibuprofen is used as an antipyretic to reduce fever in viral or bacterial infections. Additionally, it provides relief for tension-type headaches and migraines by inhibiting the production of prostaglandins that signal pain and promote inflammatory responses in the body."
    },
    {
        "id": "t03", "hop_level": 2, "category": "relationship",
        "query": "What conditions affect the kidney and have available drug treatments?",
        "reference": "Several medical conditions can impair renal function, most notably chronic kidney disease (CKD) and diabetic nephropathy. These conditions are primarily managed with ACE inhibitors or ARBs, which provide renoprotective effects by reducing intraglomerular pressure. Nephrotic syndrome, characterized by heavy protein loss in the urine, may require corticosteroids or other immunosuppressants to control the underlying inflammation. In cases of acute kidney injury due to infection, such as pyelonephritis, specific antibiotics are utilized to clear the pathogens. Furthermore, complications of kidney disease, such as hyperphosphatemia and anemia, are treated with phosphate binders and erythropoiesis-stimulating agents, respectively."
    },
    {
        "id": "t04", "hop_level": 2, "category": "side_effect",
        "query": "Which diabetes medications cause weight gain as a side effect?",
        "reference": "Weight gain is a significant side effect associated with several classes of glucose-lowering medications, which can be counterproductive in Type 2 Diabetes management. Insulin therapy frequently causes weight gain as it promotes glucose storage and inhibits lipolysis. Sulfonylureas, such as glipizide and glyburide, stimulate the pancreas to produce more insulin, often leading to increased body mass. Thiazolidinediones, like pioglitazone, are also known for causing weight gain through a combination of fluid retention and the redistribution of fat. In contrast, newer classes like GLP-1 receptor agonists and SGLT2 inhibitors are generally associated with weight loss, making them preferable for many overweight diabetic patients."
    },
    {
        "id": "t05", "hop_level": 2, "category": "relationship",
        "query": "What symptoms overlap between Asthma and Heart Failure?",
        "reference": "Asthma and congestive heart failure (CHF) frequently present with overlapping respiratory symptoms, leading to potential diagnostic challenges, particularly in elderly patients. Both conditions are characterized by dyspnea (shortness of breath), coughing, and wheezing—a phenomenon sometimes referred to as 'cardiac asthma' in the context of heart failure. However, distinguishing features do exist; heart failure is often associated with orthopnea (difficulty breathing when lying flat), paroxysmal nocturnal dyspnea, and peripheral edema (swelling of the legs). Asthma, conversely, is usually triggered by specific environmental factors and may be associated with a history of atopy or allergies. Chest X-rays and BNP blood tests are essential for differentiating the two."
    },
    {
        "id": "t06", "hop_level": 2, "category": "relationship",
        "query": "Which drugs used for depression also treat chronic pain?",
        "reference": "Several classes of antidepressants have been found to be effective in the management of chronic pain, independent of their effect on mood. Serotonin-Norepinephrine Reuptake Inhibitors (SNRIs), such as duloxetine (Cymbalta) and milnacipran, are FDA-approved for conditions like fibromyalgia and diabetic peripheral neuropathy. Tricyclic antidepressants (TCAs), specifically amitriptyline and nortriptyline, are also widely used for neuropathic pain and migraine prophylaxis. These medications work by modulating the descending inhibitory pain pathways in the central nervous system, increasing the concentration of neurotransmitters that dampen pain signals. This dual efficacy makes them invaluable for patients suffering from comorbid depression and chronic pain syndromes."
    },
    {
        "id": "t07", "hop_level": 2, "category": "treatment",
        "query": "What diseases caused by inflammation are treated with steroids?",
        "reference": "Corticosteroids, such as prednisone and methylprednisolone, are potent anti-inflammatory agents used to treat a broad spectrum of diseases characterized by an overactive immune response. They are foundational in the management of autoimmune conditions like systemic lupus erythematosus (SLE) and rheumatoid arthritis. In respiratory medicine, they are used to treat acute exacerbations of asthma and chronic obstructive pulmonary disease (COPD). Steroids are also essential for controlling inflammation in inflammatory bowel diseases (IBD), such as Crohn's disease and ulcerative colitis. Additionally, they are used in dermatology for severe eczema and psoriasis, and in oncology to reduce swelling around tumors or as part of chemotherapy regimens."
    },
    {
        "id": "t08", "hop_level": 2, "category": "side_effect",
        "query": "Which antibiotics treat respiratory infections and have kidney-related side effects?",
        "reference": "Several antibiotics used for severe or resistant respiratory infections carry a risk of nephrotoxicity. Aminoglycosides, such as gentamicin and amikacin, are highly effective against certain pathogens but require careful blood level monitoring to prevent acute tubular necrosis. Some fluoroquinolones, while generally well-tolerated, require dosage adjustments in patients with pre-existing renal impairment to avoid accumulation. Vancomycin, often used for methicillin-resistant Staphylococcus aureus (MRSA) pneumonia, is also associated with kidney injury, especially when used in combination with other nephrotoxic agents like piperacillin-tazobactam. Clinicians must balance the need for potent antimicrobial therapy with the risk of compromising the patient's renal function during the treatment of severe lung infections."
    },
    {
        "id": "t09", "hop_level": 2, "category": "relationship",
        "query": "What cardiovascular conditions are caused by Diabetes complications?",
        "reference": "Chronic diabetes mellitus leads to significant macrovascular complications that dramatically increase the risk of cardiovascular disease. The primary mechanism is accelerated atherosclerosis, which leads to coronary artery disease (CAD) and an increased incidence of myocardial infarction (heart attack). Diabetes also causes changes in the heart muscle itself, known as diabetic cardiomyopathy, which can progress to heart failure even in the absence of CAD. Furthermore, peripheral artery disease (PAD) is common, often resulting in reduced blood flow to the limbs and increasing the risk of amputations. The combination of hypertension and dyslipidemia, which frequently co-exist with diabetes, further compounds these risks, making cardiovascular health a primary focus of diabetic care."
    },
    {
        "id": "t10", "hop_level": 2, "category": "treatment",
        "query": "Which conditions treated by beta-blockers also benefit from ACE inhibitors?",
        "reference": "Beta-blockers and ACE inhibitors are frequently used together in the management of several major cardiovascular conditions due to their synergistic effects. In patients with heart failure with reduced ejection fraction (HFrEF), the combination is the standard of care to reduce mortality and improve heart function. Following a myocardial infarction (heart attack), both drugs help in cardiac remodeling and prevent future events. They are also both utilized in the long-term management of chronic hypertension, particularly when compelling indications like heart failure or diabetes are present. While beta-blockers primarily reduce heart rate and contractility, ACE inhibitors reduce systemic vascular resistance and prevent the harmful effects of the renin-angiotensin-aldosterone system."
    },

    # ── HOP LEVEL 3: Multi-hop reasoning queries ─────────────
    # Requires 3+ entity hops. Expected: GraphRAG advantage ~38%.
    # Purpose: prove the core thesis — graphs beat RAG at complexity.
    {
        "id": "h01", "hop_level": 3, "category": "multi-hop",
        "query": "Which conditions treated by ACE inhibitors have symptoms overlapping with Diabetes side effects?",
        "reference": "ACE inhibitors are primarily used for hypertension, heart failure, and chronic kidney disease. These conditions often present with fatigue and peripheral edema, symptoms that overlap significantly with diabetic complications and the side effects of certain diabetes medications. For example, diabetic nephropathy (a complication of diabetes) causes edema, which is also a sign of heart failure treated by ACE inhibitors. Additionally, the fatigue seen in heart failure can be easily confused with the lethargy caused by hypoglycemia or the side effects of metformin. This overlap necessitates a careful clinical evaluation to distinguish whether a patient's worsening symptoms are due to their underlying cardiovascular disease, a new diabetic complication, or a side effect of their therapeutic regimen."
    },
    {
        "id": "h02", "hop_level": 3, "category": "multi-hop",
        "query": "How does insulin resistance lead to cardiovascular disease in obese patients?",
        "reference": "In obese patients, insulin resistance initiates a complex pathophysiological cascade that culminates in cardiovascular disease. Excess adipose tissue, particularly visceral fat, releases pro-inflammatory cytokines and free fatty acids that impair insulin signaling. This leads to compensatory hyperinsulinemia, which stimulates the sympathetic nervous system and promotes sodium retention, contributing to hypertension. Simultaneously, insulin resistance causes an atherogenic dyslipidemia characterized by high triglycerides and low HDL cholesterol. This metabolic environment promotes oxidative stress and endothelial dysfunction, which are the foundational steps of atherosclerosis. Over time, these factors result in the development of coronary artery disease, increasing the risk of heart attacks and heart failure in this population."
    },
    {
        "id": "h03", "hop_level": 3, "category": "multi-hop",
        "query": "What drug chain connects Hypertension treatment to a side effect that causes another treatable condition?",
        "reference": "A classic clinical example of a drug-induced condition starts with the use of ACE inhibitors, such as lisinopril, for the treatment of hypertension. A well-known side effect of this class is a persistent, dry cough, which occurs in approximately 5-20% of patients due to the accumulation of bradykinin in the upper airways. This nocturnal cough can become severe enough to cause significant sleep disruption or secondary insomnia. The resulting insomnia may then require its own medical management, such as the short-term use of sedative-hypnotics like zolpidem or behavioral interventions. Thus, the treatment of one condition (hypertension) leads to a side effect (cough) that necessitates the treatment of a third condition (insomnia), highlighting the importance of medication review in complex patients."
    },
    {
        "id": "h04", "hop_level": 3, "category": "multi-hop",
        "query": "Which diseases share both a common drug treatment and a common complication affecting the heart?",
        "reference": "Type 2 Diabetes and chronic hypertension frequently share ACE inhibitors as a first-line therapeutic intervention, primarily due to their ability to protect the kidneys from damage. Both of these diseases are also major independent risk factors for the development of congestive heart failure. In patients with both conditions, the risk of heart failure is exponentially increased. Chronic high blood pressure leads to left ventricular hypertrophy and diastolic dysfunction, while diabetes contributes to myocardial fibrosis and impaired energy metabolism. Consequently, the common therapeutic pathway of using ACE inhibitors aims to mitigate the common cardiac complication of heart failure and reduce the high rate of cardiovascular mortality associated with these prevalent metabolic and vascular disorders."
    },
    {
        "id": "h05", "hop_level": 3, "category": "multi-hop",
        "query": "What conditions caused by chronic inflammation lead to treatments that have immune-suppression side effects?",
        "reference": "Chronic inflammatory diseases like rheumatoid arthritis, Crohn's disease, and systemic lupus erythematosus are driven by a dysregulated immune system that attacks the body's own tissues. To manage these conditions, potent treatments such as methotrexate, corticosteroids, and biologic agents (e.g., TNF-alpha inhibitors like adalimumab) are employed to dampen the inflammatory response. However, the primary side effect of these life-altering treatments is systemic immunosuppression. By inhibiting the immune pathways necessary for controlling inflammation, these drugs also reduce the body's ability to fight off external pathogens. This leads to a significantly increased risk of opportunistic infections, such as tuberculosis reactivation or fungal infections, requiring patients to undergo rigorous screening and vaccinations prior to and during treatment."
    },
    {
        "id": "h06", "hop_level": 3, "category": "multi-hop",
        "query": "Which pain medications treat arthritis, affect kidney function, and interact with blood pressure drugs?",
        "reference": "Non-steroidal anti-inflammatory drugs (NSAIDs), such as ibuprofen and naproxen, are commonly used to manage the pain and inflammation associated with osteoarthritis and rheumatoid arthritis. However, these medications can have significant adverse effects on renal physiology by inhibiting prostaglandins that maintain renal blood flow, potentially leading to acute kidney injury or worsening chronic kidney disease. Furthermore, NSAIDs are known to interfere with the efficacy of several classes of antihypertensive medications, including ACE inhibitors and diuretics. By causing sodium retention and reducing the vasodilatory effects of blood pressure drugs, NSAIDs can lead to a clinically significant increase in blood pressure. This triple-threat—treating arthritis while potentially damaging the kidneys and undermining hypertension control—requires careful coordination of care for elderly patients."
    },
    {
        "id": "h07", "hop_level": 3, "category": "multi-hop",
        "query": "Trace the link between obesity, insulin resistance, fatty liver disease, and available treatments.",
        "reference": "Obesity is the primary driver for the development of insulin resistance, where cells fail to respond effectively to insulin. This metabolic state leads to increased lipolysis and the shunting of free fatty acids to the liver, resulting in Non-Alcoholic Fatty Liver Disease (NAFLD). If the inflammation progresses, it can lead to Non-Alcoholic Steatohepatitis (NASH) and eventually cirrhosis. Management of this spectrum involves several levels of intervention. Treating the underlying insulin resistance with metformin remains a standard approach. Newer treatments for obesity, such as GLP-1 receptor agonists (e.g., semaglutide), have shown significant promise in reducing liver fat by promoting substantial weight loss. Additionally, Vitamin E and pioglitazone may be used in specific cases to reduce hepatic inflammation, although lifestyle-induced weight loss remains the most effective long-term treatment."
    },
    {
        "id": "h08", "hop_level": 3, "category": "multi-hop",
        "query": "What neurological conditions arise as complications of Diabetes and what drugs treat both the cause and effect?",
        "reference": "The most common neurological complication of chronic diabetes is peripheral neuropathy, caused by long-term hyperglycemia leading to nerve fiber damage and impaired blood flow to the nerves. This often manifests as distal symmetric polyneuropathy, causing burning pain or loss of sensation in the feet. To manage this condition, clinicians must address both the underlying cause and the resulting symptoms. The primary 'cause'—hyperglycemia—is treated with anti-diabetic agents like metformin, sulfonylureas, or insulin to achieve tight glycemic control. The 'effect'—neuropathic pain—is managed with specific medications like duloxetine (an SNRI) or gabapentinoids (like pregabalin), which modulate the transmission of pain signals in the system. Integrating both approaches is essential to prevent further nerve damage while improving the patient's quality of life."
    },
    {
        "id": "h09", "hop_level": 3, "category": "multi-hop",
        "query": "Which immune disorders trigger hormonal diseases that require both immunosuppressants and hormone replacement?",
        "reference": "Autoimmune endocrine disorders, such as Hashimoto's thyroiditis or Addison's disease, occur when the immune system targets and destroys hormone-producing glands. In Hashimoto's, the thyroid gland is slowly destroyed, leading to hypothyroidism. The treatment requires lifelong hormone replacement therapy with levothyroxine to maintain normal metabolic function. While hormone replacement treats the 'effect,' the underlying 'cause' is an immune disorder. In some related conditions like Graves' ophthalmopathy or certain types of autoimmune adrenalitis, corticosteroids or other immunosuppressants may be required to dampen the active autoimmune attack on the tissues. Thus, patients may find themselves in a complex balancing act, requiring medications to replace the hormones they can no longer produce while simultaneously taking drugs to suppress the very immune system that caused the destruction."
    },
    {
        "id": "h10", "hop_level": 3, "category": "multi-hop",
        "query": "What heart conditions caused by hypertension complications are treated by drugs that also affect Diabetes management?",
        "reference": "Long-standing hypertension often leads to hypertensive heart disease, which can manifest as left ventricular hypertrophy and eventually progress to heart failure with preserved or reduced ejection fraction. The medications used to treat these heart conditions, specifically ACE inhibitors and certain beta-blockers, have significant implications for diabetes management. ACE inhibitors are the preferred treatment for hypertensive heart failure because they also offer significant protection against diabetic nephropathy by reducing renal efferent arteriolar resistance. Conversely, some non-selective beta-blockers can mask the symptoms of hypoglycemia (like tremors and tachycardia) in diabetic patients, potentially making insulin therapy more dangerous. Therefore, selecting the right cardiovascular drugs for a patient with both hypertension-related heart disease and diabetes is a critical clinical decision to maximize organ protection while minimizing metabolic risks."
    }
]

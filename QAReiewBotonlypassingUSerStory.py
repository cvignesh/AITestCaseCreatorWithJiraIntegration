import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from urllib.parse import urljoin
import re

# --- JIRA CONFIGURATION ---
JIRA_CONFIG = {
    'JIRA_URL': "Jira URL",
    'JIRA_USERNAME': "<Jira User Name>"
    'JIRA_PASSWORD': "<Jira Password>",
    'TEST_ISSUE_TYPE': "Test"
}


# --- LLM FUNCTIONS ---
def call_openai(api_key, prompt):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a QA test expert."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def call_ollama(endpoint, model, prompt):
    response = requests.post(
        f"{endpoint}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )
    return response.json()["message"]["content"].strip()


def call_groq(endpoint, api_key, model, prompt):
    headers = {"Authorization": f"Bearer {api_key}"}
    json_data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature":0.2
    }
    response = requests.post(endpoint, headers=headers, json=json_data)
    return response.json()["choices"][0]["message"]["content"].strip()


# --- JIRA FUNCTIONS ---
def get_jira_test_steps(test_key):
    """Fetch test steps for a given test case"""
    url = urljoin(JIRA_CONFIG['JIRA_URL'], f"/rest/api/2/issue/{test_key}")
    auth = HTTPBasicAuth(JIRA_CONFIG['JIRA_USERNAME'], JIRA_CONFIG['JIRA_PASSWORD'])

    try:
        response = requests.get(url, auth=auth)
        response.raise_for_status()
        test_data = response.json()

        # Extract steps from Xray custom field (adjust field ID as needed)
        steps = []
        if 'customfield_10115' in test_data['fields'] and 'steps' in test_data['fields']['customfield_10115']:
            for step in test_data['fields']['customfield_10115']['steps']:
                steps.append({
                    'index': step['index'],
                    'action': step['fields'].get('Action', ''),
                    'data': step['fields'].get('Data', ''),
                    'expected': step['fields'].get('Expected Result', '')
                })
        return steps
    except Exception as e:
        st.error(f"Error fetching test steps: {str(e)}")
        return []


def get_jira_issue_details(issue_key):
    url = urljoin(JIRA_CONFIG['JIRA_URL'], f"/rest/api/2/issue/{issue_key}?expand=issuelinks")
    auth = HTTPBasicAuth(JIRA_CONFIG['JIRA_USERNAME'], JIRA_CONFIG['JIRA_PASSWORD'])

    try:
        response = requests.get(url, auth=auth)
        response.raise_for_status()
        issue_data = response.json()

        # Extract project key from the issue
        project_key = issue_data['fields']['project']['key']

        # Extract linked tests with steps
        linked_tests = []
        for link in issue_data['fields']['issuelinks']:
            test = None
            if 'outwardIssue' in link and link['outwardIssue']['fields']['issuetype']['name'] == 'Test':
                test = link['outwardIssue']
            elif 'inwardIssue' in link and link['inwardIssue']['fields']['issuetype']['name'] == 'Test':
                test = link['inwardIssue']

            if test:
                steps = get_jira_test_steps(test['key'])
                steps_formatted = "\n".join(
                    f"{step['index']}. {step['action']}" +
                    (f"\n   Data: {step['data']}" if step['data'] else "") +
                    (f"\n   Expected: {step['expected']}" if step['expected'] else "")
                    for step in steps
                ) if steps else "No steps defined"

                linked_tests.append({
                    'Key': test['key'],
                    'Summary': test['fields']['summary'],
                    'Status': test['fields']['status']['name'],
                    'Steps': steps_formatted,
                    'Expected Result': "\n".join(
                        step['expected'] for step in steps if step.get('expected')) if steps else ""
                })

        return {
            'key': issue_data['key'],
            'summary': issue_data['fields']['summary'],
            'description': issue_data['fields'].get('description', ''),
            'project_key': project_key,
            'linked_tests': linked_tests
        }
    except Exception as e:
        st.error(f"Error fetching Jira issue: {str(e)}")
        return None


def create_jira_test_case(test_data, user_story_details):
    url = urljoin(JIRA_CONFIG['JIRA_URL'], "/rest/api/2/issue")
    auth = HTTPBasicAuth(JIRA_CONFIG['JIRA_USERNAME'], JIRA_CONFIG['JIRA_PASSWORD'])
    headers = {"Content-Type": "application/json"}

    steps = [step.strip() for step in test_data['Steps'].split('\n') if step.strip()]

    test_issue = {
        "fields": {
            "project": {"key": user_story_details['project_key']},
            "issuetype": {"name": JIRA_CONFIG['TEST_ISSUE_TYPE']},
            "summary": test_data['Title'],
            "description": f"Steps:\n{test_data['Steps']}\n\nExpected Result:\n{test_data['Expected Result']}",
            "customfield_10115": {
                "steps": [{
                    "index": i + 1,
                    "fields": {
                        "Action": step,
                        "Data": "",
                        "Expected Result": test_data['Expected Result']
                    }
                } for i, step in enumerate(steps)]
            }
        }
    }

    try:
        response = requests.post(url, auth=auth, headers=headers, json=test_issue)
        response.raise_for_status()
        test_key = response.json().get('key')

        # Link to user story
        link_data = {
            "type": {"name": "Tests"},
            "inwardIssue": {"key": test_key},
            "outwardIssue": {"key": user_story_details['key']}
        }
        requests.post(
            urljoin(JIRA_CONFIG['JIRA_URL'], "/rest/api/2/issueLink"),
            auth=auth,
            headers=headers,
            json=link_data
        )
        return test_key
    except Exception as e:
        st.error(f"Error creating test: {str(e)}")
        return None


# --- PARSER ---
def parse_llm_response(response):
    titles = re.findall(r"(?:Title|Test Case):\s*(.*)", response)
    steps_raw = re.findall(r"Steps:\n((?:- .+\n?)+)", response)
    expected = re.findall(r"Expected Result:\s*(.*)", response)

    rows = []
    for i in range(len(titles)):
        title = titles[i].strip()
        steps_list = re.findall(r"- (.+)", steps_raw[i]) if i < len(steps_raw) else []
        steps = "\n".join(f"{j + 1}. {step}" for j, step in enumerate(steps_list))
        result = expected[i].strip() if i < len(expected) else ""
        rows.append({
            "Include": True,
            "Title": title,
            "Steps": steps,
            "Expected Result": result,
        })
    return pd.DataFrame(rows)


# --- STREAMLIT APP ---
st.set_page_config(page_title="Jira Test Case Generator", layout="wide")
st.title("🧪 Jira Test Case Generator")

# Initialize session state
if 'suggested_tests' not in st.session_state:
    st.session_state.suggested_tests = pd.DataFrame()

# Sidebar - LLM Configuration
with st.sidebar:
    st.header("🔧 LLM Settings")
    llm_choice = st.selectbox("Choose LLM", ["OpenAI", "Ollama", "Groq"])

    if llm_choice == "OpenAI":
        openai_key = st.text_input("OpenAI API Key", type="password")
    elif llm_choice == "Ollama":
        ollama_endpoint = st.text_input("Ollama Endpoint", value="http://localhost:11434")
        ollama_model = st.text_input("Ollama Model", value="llama2")
    elif llm_choice == "Groq":
        groq_endpoint = st.text_input("Groq Endpoint", value="https://api.groq.com/openai/v1/chat/completions")
        groq_key = st.text_input("Groq API Key", type="password")
        groq_model = st.text_input("Groq Model", value="mixtral-8x7b-32768")

# Main Content
issue_key = st.text_input("🔍 Enter Jira User Story Key (e.g., AIP-123):", "").strip().upper()

if issue_key:
    with st.spinner("Fetching user story details from Jira..."):
        user_story = get_jira_issue_details(issue_key)

        if user_story:
            st.success(f"Fetched user story: {user_story['key']}")

            # Display User Story Details
            st.markdown("### 📋 User Story Details")
            st.markdown(f"**Key:** `{user_story['key']}`")
            st.markdown(f"**Summary:** {user_story['summary']}")
            st.markdown(f"**Description:**")
            st.markdown(user_story['description'] or "*No description*")

            # Display Linked Tests in Table Format
            if user_story['linked_tests']:
                st.markdown("### 🔗 Linked Test Cases")
                linked_tests_df = pd.DataFrame(user_story['linked_tests'])

                # Convert to display-friendly format
                display_df = linked_tests_df[['Key', 'Summary', 'Status', 'Steps', 'Expected Result']]
                display_df.columns = ['Test Key', 'Summary', 'Status', 'Steps', 'Expected Result']

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Test Key": st.column_config.TextColumn(width="small"),
                        "Summary": st.column_config.TextColumn(width="medium"),
                        "Status": st.column_config.TextColumn(width="small"),
                        "Steps": st.column_config.TextColumn(width="large"),
                        "Expected Result": st.column_config.TextColumn(width="medium")
                    }
                )
            else:
                st.info("No existing test cases linked to this user story")

            # Add new text box for additional instructions
            additional_instructions = st.text_area(
                "📝 Additional Instructions for Test Case Generation",
                placeholder="Enter any specific requirements, edge cases, or additional scenarios you want to include in the test cases...",
                height=100
            )

            # Generate Suggested Test Cases
            if st.button("💡 Generate Suggested Test Cases"):
                if llm_choice == "OpenAI" and not openai_key:
                    st.error("OpenAI API key required")
                elif llm_choice == "Ollama" and not ollama_endpoint:
                    st.error("Ollama endpoint required")
                elif llm_choice == "Groq" and not all([groq_endpoint, groq_key, groq_model]):
                    st.error("Groq configuration incomplete")
                else:
                    with st.spinner("Generating test case suggestions..."):
                        prompt = f"""
                        As a QA expert, suggest comprehensive test cases for this user story:

                        User Story: {user_story['summary']}
                        Description: {user_story['description']}

                        {f"Additional Instructions: {additional_instructions}" if additional_instructions else ""}

                        Include positive, negative, and edge cases. Format each as:
                        Title: <test case title>
                        Steps:
                        - step 1
                        - step 2
                        Expected Result: <expected outcome>
                        """

                        try:
                            if llm_choice == "OpenAI":
                                response = call_openai(openai_key, prompt)
                            elif llm_choice == "Ollama":
                                response = call_ollama(ollama_endpoint, ollama_model, prompt)
                            elif llm_choice == "Groq":
                                response = call_groq(groq_endpoint, groq_key, groq_model, prompt)

                            st.session_state.suggested_tests = parse_llm_response(response)
                        except Exception as e:
                            st.error(f"LLM Error: {str(e)}")

# Display and Edit Suggested Tests
if not st.session_state.suggested_tests.empty:
    st.markdown("### 🧪 Suggested Test Cases")
    st.markdown("Edit and approve tests to be created in Jira:")

    # Edit suggested tests
    edited_df = st.data_editor(
        st.session_state.suggested_tests,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Include": st.column_config.CheckboxColumn("Create in Jira", default=True),
            "Title": "Test Case Title",
            "Steps": "Test Steps",
            "Expected Result": "Expected Result"
        },
        num_rows="dynamic"
    )

    # Push to Jira button
    if st.button("🚀 Push Selected Tests to Jira", type="primary"):
        selected_tests = edited_df[edited_df["Include"] == True]

        if selected_tests.empty:
            st.warning("No test cases selected!")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            created_tests = []

            for i, (_, test_case) in enumerate(selected_tests.iterrows()):
                status_text.text(f"Creating test {i + 1}/{len(selected_tests)}...")
                test_key = create_jira_test_case(test_case, user_story)
                if test_key:
                    created_tests.append(test_key)
                progress_bar.progress((i + 1) / len(selected_tests))

            progress_bar.empty()
            status_text.empty()

            if created_tests:
                st.success(f"✅ Created {len(created_tests)} test cases in Jira!")
                st.markdown("**Created Tests:**")
                for key in created_tests:
                    st.markdown(f"- `{key}`: {urljoin(JIRA_CONFIG['JIRA_URL'], f'/browse/{key}')}")
                st.balloons()
            else:
                st.error("❌ Failed to create any test cases")

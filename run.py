"""Interactive CLI: the colon-command protocol and the turn loop.

The engine (session.QwenNPU) owns history, budget and generation;
this module turns stdin into turns. Enter or Ctrl-C stops an
answer mid-stream and the partial answer is kept; Ctrl-C at the
prompt exits, as does the stdin-close key (Ctrl-D, Ctrl-Z on
Windows).
"""

from session import EOF, QwenNPU, system_messages


def handle_command(session, text):
    """Map a colon command to the outcome it requests.

    "exit" ends the session, "command" means the command was
    handled, None means the text is an ordinary message.
    """
    cmd = text.lower().strip()
    if cmd == ":exit":
        return "exit"
    if cmd == ":clear":
        session.messages = system_messages()
        return "command"
    if cmd == ":think":
        print(f"Thinking: {session.toggle_thinking()}.")
        return "command"
    if cmd == ":save":
        fname = session.save_transcript()
        print(f"Saved to {fname}")
        return "command"
    if cmd == ":help":
        print_help()
        return "command"
    return None


def print_help():
    print("Commands:")
    print("  :think - toggle thinking mode")
    print("  :save  - export the transcript")
    print("  :clear - reset the conversation")
    print("  :exit  - close")
    print("Enter or Ctrl-C stops an answer mid-stream; the partial")
    print("answer is kept and the session continues. Ctrl-C at the")
    print("prompt exits.")


def run(session):
    while True:
        try:
            print("\n---\nYou> ", end="", flush=True)
            line = session.lines.get()
            if line is EOF:
                break
            user_input = line.strip()
            if not user_input:
                continue

            action = handle_command(session, user_input)
            if action == "exit":
                break
            if action == "command":
                continue

            session.messages.append(
                {"role": "user", "content": user_input})
            prompt = session.trim_to_prompt_budget()
            # A turn longer than the slot survives trimming only
            # by deleting the user's own message, so decline the
            # turn instead; history stays aligned and trimmable.
            if not session.sendable(prompt):
                session.messages.pop()
                print("Message too long for the prompt slot;"
                      " not sent.")
                continue

            print("\n---\nAgent> ", end="", flush=True)
            session.record_answer(session.generate(prompt))
            if session.eof_seen:
                break

        except KeyboardInterrupt:
            break


def main():
    session = QwenNPU()
    session.setup()
    session.start_stdin()
    print("Ready on NPU. Type :help for commands.")
    run(session)


if __name__ == "__main__":
    main()

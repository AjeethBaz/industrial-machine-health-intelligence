import { useState } from "react";

interface MachineSelectorProps {
  onSelect: (udi: number) => void;
}

export default function MachineSelector({
  onSelect,
}: MachineSelectorProps) {
  const [udi, setUdi] = useState("");

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const parsedUdi = Number(udi);

    if (!Number.isInteger(parsedUdi) || parsedUdi <= 0) {
      return;
    }

    onSelect(parsedUdi);
  }

  return (
    <form className="machine-selector" onSubmit={handleSubmit}>
      <input
        type="number"
        min="1"
        step="1"
        placeholder="UDI e.g. 7012"
        value={udi}
        onChange={(event) => setUdi(event.target.value)}
      />
      <button type="submit">View Machine</button>
    </form>
  );
}
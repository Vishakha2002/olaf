import {
  Streamlit,
  ComponentProps,
  withStreamlitConnection,
} from "streamlit-component-lib";
import React, { useEffect, useState } from "react";

import ReactPlayer from "react-player";
import HeightObserver from "./height-observer";
import StAudioRec from "./StreamlitAudioRecorder";

const StreamlitPlayer = ({ args }: ComponentProps) => {
  const [playerEvents, setPlayerEvents] = useState({});
  const [isPlaying, setIsPlaying] = useState(false); // handling state of play/pause of player
  const playerRef = React.useRef<ReactPlayer>(null);
  const divRef = React.useRef<HTMLDivElement>(null);

  // Handle events
  useEffect(() => {
    divRef.current?.focus();
    let events: any = {};

    window.removeEventListener("keypress", handleKeypress);
    window.addEventListener("keypress", handleKeypress);

    args.events.forEach((name: string) => {
      events[name] = (data?: any) => {
        Streamlit.setComponentValue({
          name: name,
          data: data,
        });
      };
    });

    setPlayerEvents(events);
  }, [args.events]);

  const onKeyPressHandler = () => {
    setIsPlaying(!isPlaying);
  };

  const handleKeypress = (event: { key: string }) => {
    if (event.key === "p") {
      // Video Play/pause toggle using p letter on keyboard
      onKeyPressHandler();
    }
    if (event.key === "a") {
      // Video needs to pause and frame captured when question is being asked
      setIsPlaying(false);
    }
  };

  return (
    <div ref={divRef} id="container" style={{ height: "400px" }}>
      <br></br>
      <br></br>
      <HeightObserver
        onChange={Streamlit.setFrameHeight}
        fixedHeight={args.height}
      >
        <ReactPlayer
          ref={playerRef}
          url={args.url}
          width={args.width || undefined}
          height={args.height || undefined}
          playing={isPlaying || undefined}
          loop={args.loop || undefined}
          controls={args.controls || undefined}
          light={args.light || undefined}
          volume={args.volume}
          muted={args.muted || undefined}
          playbackRate={args.playbackRate}
          progressInterval={args.progressInterval}
          playsinline={args.playInline || undefined}
          config={args.config || undefined}
          {...playerEvents}
        />
      </HeightObserver>
      <br></br>
      <br></br>
      <StAudioRec />
    </div>
  );
};

export default withStreamlitConnection(StreamlitPlayer);

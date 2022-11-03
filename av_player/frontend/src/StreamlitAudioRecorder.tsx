import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib";
import React, { ReactNode } from "react";
import "./StreamlitAudioRecorder.css";
import AudioReactRecorder, { RecordState } from "audio-react-recorder";
import "audio-react-recorder/dist/index.css";

interface State {
  isFocused: boolean;
  recordState: null;
  audioDataURL: string;
  reset: boolean;
}

class StAudioRec extends StreamlitComponentBase<State> {

  public state = {
    isFocused: true,
    recordState: null,
    audioDataURL: "",
    reset: false
  };

  public render = (): ReactNode => {

    // Arguments that are passed to the plugin in Python are accessible
    // Streamlit sends us a theme object via props that we can use to ensure
    // that our component has visuals that match the active theme in a
    // streamlit app.

    const { theme } = this.props;
    const style: React.CSSProperties = {};

    const { recordState } = this.state;

    const handleKeypress = (event: { key: string }) => {
      if (event.key === "a") {
        // start audio recording
        this.onClick_start();
      }
      if (event.key === "f") {
        // stop audio recording and hide component as well
        this.onClick_stop();
      }
      if (event.key === "p") {
        // stop audio recording and hide component as well
        this.onClick_stop();
      }
    };
    window.removeEventListener("keypress", handleKeypress);
    window.addEventListener("keypress", handleKeypress);

    // compatibility with older vers of Streamlit that don't send theme object.
    if (theme) {
      // Use the theme object to style our button border. Alternatively, the
      // theme style is defined in CSS vars.
      const borderStyling = `1px solid ${
        this.state.isFocused ? theme.primaryColor : "gray"
      }`;
      style.border = borderStyling;
      style.outline = borderStyling;
    }

    return (
      <span>
        <div>
          <div id="outer">
            <div className="inner">
              <button id="record" onClick={this.onClick_start}>
                Click to Ask!
              </button>
            </div>
            <div className="inner">
              <button id="stop" onClick={this.onClick_stop}>
                Fetch response!
              </button>
            </div>
          </div>
          <AudioReactRecorder
            state={recordState}
            onStop={this.onStop_audio}
            type="audio/wav"
            backgroundColor="rgb(255, 255, 255)"
            foregroundColor="rgb(255,76,75)"
            canvasWidth={450}
            canvasHeight={100}
          />
        </div>
      </span>
    );
  };

  onClick_start = () => {
    const audioElement = document.querySelectorAll<HTMLElement>('.audio-react-recorder');
    audioElement[0].style.visibility = 'visible';
    this.setState({
      reset: false,
      audioDataURL: "",
      recordState: RecordState.START,
    });
    Streamlit.setComponentValue("");
  };

  private onClick_stop = () => {
    this.setState({
      reset: false,
      recordState: RecordState.STOP,
      isFocused: false
    });
    const audioElement = document.querySelectorAll<HTMLElement>('.audio-react-recorder');
    audioElement[0].style.visibility = 'hidden';
  };

  private onStop_audio = (data) => {
    if (this.state.reset === true) {
      this.setState({
        audioDataURL: "",
      });
      Streamlit.setComponentValue("");
    } else {
      this.setState({
        audioDataURL: data.url,
      });
      fetch(data.url)
        .then(function (ctx) {
          return ctx.blob();
        })
        .then(function (blob) {
          // converting blob to arrayBuffer, this process step needs to be be improved
          // this operation's time complexity scales exponentially with audio length
          return new Response(blob).arrayBuffer();
        })
        .then(function (buffer) {
          console.log("setting setComponentValue");
          Streamlit.setComponentValue({
            arr: new Uint8Array(buffer),
          });
        });
    }
  };
}

// "withStreamlitConnection" is a wrapper function. It bootstraps the
// connection between your component and the Streamlit app, and handles
// passing arguments from Python -> Component.
// You don't need to edit withStreamlitConnection (but you're welcome to!).
export default withStreamlitConnection(StAudioRec);

// Tell Streamlit we're ready to start receiving data. We won't get our
// first RENDER_EVENT until we call this function.
Streamlit.setComponentReady();

// Finally, tell Streamlit to update our initial height. We omit the
// `height` parameter here to have it default to our scrollHeight.
Streamlit.setFrameHeight();

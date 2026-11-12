/**
 * The long help that stood on the custom node cards, under their fields in small print: taken off the cards (a card
 * says what it does and holds what is set; paragraphs of help made it a page) and shown in the Properties panel for
 * the chosen node, under "About this node", after the node's own one-line description. By the node's type.
 *
 * Each names the field it is about, since it no longer stands under it. A card keeps a note of 72 characters or
 * fewer under its field, as a generic card does (vendor/oaiy-ui-components GenericNode's CARD_HELP_MOST).
 */
export const CARD_HELP: Record<string, string[]> = {
  ai_llm: [
    'Vision Projector (mmproj): needed when a picture is sent to a model that takes pictures (a load error names what a model lacks); a model that takes none refuses the picture. Left blank, the projector that comes with the model is used.',
  ],
  database: [
    'Stores what comes in on Data (any JSON) in this collection, kept between runs.',
  ],
  database_query: [
    'Gives the records in this collection that match the filter: the one connected to Filter, else this one.',
  ],
  image_save: [
    'Each picture is saved as it arrives, while the flow runs, and the node gives on where the file went.',
    'File Path: a whole path, like /Users/you/Pictures/lake.png (the format’s ending is added if it has none).',
  ],
  input_folder: [
    'Gives the list of the files it finds: each one’s path, name, extension and size.',
  ],
  input_video: [
    'Gives the video’s file to the next step: Video Frames, Speech to Text, Video Save and the like.',
  ],
  video_frame_extractor: [
    'Batch Size: 0 takes every frame at once; more, that many at a time (less memory).',
  ],
  video_save: [
    'Each video is saved as it arrives, while the flow runs, and the node gives on where the file went.',
  ],
};
